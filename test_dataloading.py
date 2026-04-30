import os
import json
import random
import csv
import subprocess
import torch
import torchvision
from torchvision.transforms import v2
from einops import rearrange
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm

_DECODE_BACKEND_CACHE = None
_BACKEND_HINT_PRINTED = False

# ---------- FFmpeg GPU 解码 (NVDEC/CUDA) ----------
def _resolve_tool_bin(env_key, default_bin):
    """Resolve external tool path from env var with sane fallback."""
    v = os.environ.get(env_key, "").strip()
    return v if v else default_bin


def _ffmpeg_gpu_video_info(path):
    """用 ffprobe 获取视频信息。返回 dict: width, height, fps, num_frames。失败抛错。"""
    ffprobe_bin = _resolve_tool_bin("FFPROBE_BIN", "ffprobe")
    cmd = [
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
        data = json.loads(out.decode("utf-8"))
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"ffprobe failed for {path}: {e}") from e
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        raise RuntimeError(f"ffprobe: no video stream in {path}")
    s = streams[0]
    width = int(s.get("width", 0))
    height = int(s.get("height", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"ffprobe: invalid size {width}x{height} in {path}")
    # fps: "24/1" or "30000/1001"
    r = s.get("r_frame_rate", "24/1")
    if "/" in r:
        num, den = r.split("/", 1)
        fps = float(num) / float(den) if float(den) != 0 else 24.0
    else:
        fps = float(r) if r else 24.0
    nb_frames = s.get("nb_frames")
    if nb_frames is not None:
        try:
            num_frames = int(nb_frames)
        except (ValueError, TypeError):
            num_frames = None
    else:
        num_frames = None
    if num_frames is None or num_frames <= 0:
        duration = float(fmt.get("duration", 0) or 0)
        if duration > 0 and fps > 0:
            num_frames = int(round(duration * fps))
        else:
            num_frames = 0
    return {"width": width, "height": height, "fps": fps, "num_frames": num_frames}


def _decode_mode_from_env():
    mode = os.environ.get("FFMPEG_DECODE_BACKEND", "auto").strip().lower()
    if mode not in ("auto", "cuda", "cpu"):
        mode = "auto"
    return mode


def _prefer_cpu_decode_for_current_gpu():
    """A100/A800 generally lack NVDEC, so CPU decode is often the fastest supported path."""
    if os.environ.get("FFMPEG_FORCE_CPU_DECODE", "0").strip() in ("1", "true", "True"):
        return True
    try:
        if not torch.cuda.is_available():
            return True
        names = []
        for i in range(torch.cuda.device_count()):
            with suppress(Exception):
                names.append(str(torch.cuda.get_device_name(i)).upper())
        for name in names:
            if ("A100" in name) or ("A800" in name):
                return True
    except Exception:
        pass
    return False


def _decode_backend_order():
    global _DECODE_BACKEND_CACHE
    mode = _decode_mode_from_env()
    if mode in ("cuda", "cpu"):
        return [mode]

    # Auto mode: prefer the last successful backend first.
    if _DECODE_BACKEND_CACHE in ("cuda", "cpu"):
        if _DECODE_BACKEND_CACHE == "cuda":
            return ["cuda", "cpu"]
        return ["cpu", "cuda"]

    if _prefer_cpu_decode_for_current_gpu():
        return ["cpu", "cuda"]
    return ["cuda", "cpu"]


def _ffmpeg_decode_frames_with_backend(path, frame_indices, width, height, fps, backend):
    if backend not in ("cuda", "cpu"):
        raise ValueError(f"invalid decode backend: {backend}")

    start_frame = min(frame_indices)
    end_frame = max(frame_indices)
    num_decode = end_frame - start_frame + 1
    start_time_sec = start_frame / fps if fps > 0 else 0.0

    ffmpeg_bin = _resolve_tool_bin("FFMPEG_BIN", "ffmpeg")
    threads = os.environ.get("FFMPEG_CPU_THREADS", "0").strip()
    cmd = [ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error"]
    if backend == "cuda":
        cmd += ["-hwaccel", "cuda"]
    else:
        # CPU decode: auto threads by default; can override with FFMPEG_CPU_THREADS
        cmd += ["-threads", threads]

    cmd += [
        "-ss", str(start_time_sec),
        "-i", path,
        "-vframes", str(num_decode),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "pipe:1",
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=60)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(
            f"FFmpeg {backend} decode failed via {ffmpeg_bin}. stderr: {stderr[:800]}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg not found ({ffmpeg_bin}). Set FFMPEG_BIN or install ffmpeg."
        ) from None

    n = len(out) // (width * height * 3)
    if n != num_decode:
        n = min(n, num_decode)
    frames = np.frombuffer(out, dtype=np.uint8)
    frames = frames[: n * width * height * 3].reshape((n, height, width, 3))
    index_in_decode = [i - start_frame for i in frame_indices]
    result = []
    for idx in index_in_decode:
        if 0 <= idx < frames.shape[0]:
            result.append(frames[idx])
        else:
            result.append(frames[-1] if frames.shape[0] > 0 else np.zeros((height, width, 3), dtype=np.uint8))
    return np.stack(result, axis=0)


def _ffmpeg_gpu_decode_frames(path, frame_indices, width, height, fps):
    """
    用 ffmpeg -hwaccel cuda 解码指定帧，输出 RGB。要求系统 ffmpeg 带 CUDA/NVDEC。
    frame_indices: list[int]，帧下标（从 0 开始）。
    返回: np.ndarray uint8 shape (N, H, W, 3)，失败抛错。
    """
    if not frame_indices:
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    global _DECODE_BACKEND_CACHE, _BACKEND_HINT_PRINTED

    mode = _decode_mode_from_env()
    backends = _decode_backend_order()
    errs = []

    for backend in backends:
        try:
            out = _ffmpeg_decode_frames_with_backend(path, frame_indices, width, height, fps, backend)
            _DECODE_BACKEND_CACHE = backend
            if mode == "auto" and (backend == "cpu") and (not _BACKEND_HINT_PRINTED):
                print("[Decode] Auto backend selected CPU decode (compatible with A100/A800).", flush=True)
                _BACKEND_HINT_PRINTED = True
            return out
        except Exception as e:
            errs.append(f"{backend}: {e}")

    raise RuntimeError(
        "FFmpeg decode failed for all backends (auto fallback enabled). "
        + " | ".join(errs[:2])
    )

# Mock args for testing
class MockArgs:
    def __init__(self):
        self.dataset_path = "/data/server_user/Video_dataset/MovieBench_data/Processed_Video"
        self.num_frames = 81
        self.height = 480
        self.width = 832
        self.use_first_aug = False
        self.dataloader_num_workers = 4

class TextVideoDataset_New(Dataset):
    def __init__(self, base_path, args=None):
        self.args = args
        self.max_num_frames = args.num_frames
        self.num_frames = args.num_frames
        self.height = args.height
        self.width = args.width
        self.misc_size = [self.height, self.width]
        self.video_list = []
        
        # 不再使用 use_QWen.txt / blackwhite_video_folders.txt / failed_frame_extraction.txt 过滤
        print(f"Loading videos from specified path: {base_path}")
        
        # Traverse directories: Duration -> VideoFolder
        if os.path.isdir(base_path):
            categories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            for category in categories:
                cat_path = os.path.join(base_path, category)
                video_folders = [d for d in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, d))]
                
                for folder in video_folders:
                    folder_path = os.path.join(cat_path, folder)
                    # Check for required files
                    json_path = os.path.join(folder_path, "rewrite_caption.json")
                    video_file = os.path.join(folder_path, "preprocessed_video.mp4")
                    
                    if os.path.exists(json_path) and os.path.exists(video_file):
                        # Pre-check: Skip videos with only 1 chunk (after ignoring last one, there will be 0 valid chunks)
                        try:
                            with open(json_path, 'r') as f:
                                data_json = json.load(f)
                            
                            # Handle both dict and list formats
                            if isinstance(data_json, list):
                                chunks = data_json
                            elif isinstance(data_json, dict):
                                chunks = data_json.get('chunks', [])
                            else:
                                chunks = []
                            
                            # After ignoring last chunk, need at least 1 valid chunk
                            if len(chunks) <= 1:
                                # print(f"Skipping {folder_path}: Only {len(chunks)} chunk(s), need at least 2 chunks")
                                continue
                        except Exception as e:
                            print(f"Warning: Could not pre-check chunks for {folder_path}: {e}, will check during loading")
                        
                        self.video_list.append({
                            'path': folder_path,
                            'json_path': json_path,
                            'video_path': video_file
                        })
        
        # 3. Apply sample list filtering if specified
        sample_list_path = getattr(args, 'sample_list_path', None) if args else None
        if sample_list_path and os.path.exists(sample_list_path):
            with open(sample_list_path, 'r') as f:
                selected_samples = json.load(f)
            selected_folders = {s['folder'] for s in selected_samples}
            
            filtered_list = []
            for item in self.video_list:
                path = item.get('path', '')
                if path:
                    folder_name = path.split('/')[-1]
                    if folder_name in selected_folders:
                        filtered_list.append(item)
            self.video_list = filtered_list
            print(f"Applied sample list filter: {len(self.video_list)} samples after filtering from {sample_list_path}")
        
        print(f"Total videos loaded: {len(self.video_list)}")
        
        self.frame_process = v2.Compose([
            v2.Resize(size=(self.height, self.width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def resize(self, image):
        image = torchvision.transforms.functional.resize(
            image,
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image
    
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(
            image,
            (self.height, self.width),
        )
        return image

    def load_video_segment(self, video_path, start_time, end_time, target_num_frames=81):
        """使用 FFmpeg 自动解码（优先可用最快后端，CUDA 失败自动回退 CPU）。"""
        info = _ffmpeg_gpu_video_info(video_path)
        total_frames = info["num_frames"]
        fps = info["fps"]
        w, h = info["width"], info["height"]
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        if end_frame > total_frames:
            raise ValueError(f"Annotated end frame {end_frame} exceeds total frames {total_frames} in {video_path}")
        segment_len = end_frame - start_frame
        if segment_len <= 0:
            raise ValueError(f"Invalid segment length {segment_len} (start: {start_frame}, end: {end_frame}) in {video_path}")
        if segment_len < target_num_frames:
            raise ValueError(f"Segment length {segment_len} frames is shorter than required {target_num_frames} frames in {video_path}")
        stride = max(segment_len // target_num_frames, 1)
        frame_indices = []
        current_idx = start_frame
        while len(frame_indices) < target_num_frames:
            if current_idx >= end_frame:
                frame_indices.append(frame_indices[-1] if frame_indices else start_frame)
            else:
                frame_indices.append(current_idx)
                current_idx += stride
        safe_indices = [min(i, total_frames - 1) for i in frame_indices]
        batch = _ffmpeg_gpu_decode_frames(video_path, safe_indices, w, h, fps)  # (N, H, W, 3)
        frames = []
        for i in range(batch.shape[0]):
            frame = Image.fromarray(batch[i]).convert("RGB")
            frame = self.crop_and_resize(frame)
            frame = self.frame_process(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        return rearrange(frames, "T C H W -> C T H W")

    def process_frames_for_ref(self, video_tensor):
        # Video tensor is C T H W
        # Original: first_ref_frames (list of first 12 frames), random_ref_frame (tensor)
        # Note: Original code did standard loading then "first_ref_frames = [frame_list[idx].copy() ...]" before tensor conversion
        # Here we already have tensors.
        
        C, T, H, W = video_tensor.shape
        
        # Simulate original: first 12 frames
        # Original code applied crop_and_resize then frame_process. 
        # My load_video_segment applies crop_and_resize then frame_process.
        # So I can just slice the tensor.
        
        num_ref = min(12, T)
        first_ref_frames = [video_tensor[:, i, :, :].clone() for i in range(num_ref)]
        # If less than 12, pad? Original code "ref_frame_indices = list(range(num_ref_frames))" assumes T >= 12?
        # Original max_num_frames=81.
        # If I strictly enforced T=81, then this is fine.
        
        random_idx = random.randint(0, T - 1)
        random_ref_frame = video_tensor[:, random_idx, :, :].clone()
        
        return first_ref_frames, random_ref_frame

    def check_needs_memory_extraction(self, index):
        """
        Lightweight check to determine if a sample needs memory extraction.
        Only reads JSON file, does not load video.
        Simulates the selection logic from __getitem__ to determine if memory could be used.
        Returns: (needs_extraction: bool, metadata_dict or None)
        """
        item = self.video_list[index % len(self.video_list)]
        folder_path = item['path']
        json_path = item['json_path']
        
        try:
            with open(json_path, 'r') as f:
                data_json = json.load(f)
            
            # Handle both dict and list formats
            if isinstance(data_json, list):
                chunks = data_json
            elif isinstance(data_json, dict):
                chunks = data_json.get('chunks', [])
            else:
                return False, None
            
            # Ignore last chunk
            if len(chunks) > 0:
                valid_chunks = chunks[:-1]
            else:
                valid_chunks = []
            
            if len(valid_chunks) == 0:
                return False, None
            
            # Check if memory extraction is possible
            if len(valid_chunks) == 1:
                # Only one valid chunk, no memory possible
                return False, None
            
            # Simulate selection logic: check if ANY core chunk (from valid_chunks[1:]) 
            # could find a matching memory chunk (from valid_chunks[0:core_idx])
            # We check multiple possible core_idx values to see if any would use memory
            
            for test_core_idx in range(1, len(valid_chunks)):
                core_chunk = valid_chunks[test_core_idx]
                core_chars = core_chunk.get('character_list', [])
                
                if not core_chars or len(core_chars) == 0:
                    continue
                
                # Check if there's any memory chunk before this core_idx that matches
                possible_memory_indices = list(range(0, test_core_idx))
                
                for m_idx in possible_memory_indices:
                    m_chunk = valid_chunks[m_idx]
                    m_chars_list = m_chunk.get('character_list', [])
                    
                    if not m_chars_list or len(m_chars_list) == 0:
                        continue
                    
                    m_chars = set(m_chars_list)
                    core_chars_set = set(core_chars)
                    
                    # Check if there's any intersection (at least one character matches)
                    if core_chars_set & m_chars:
                        # This sample could use memory
                        folder_name = os.path.basename(folder_path)
                        return True, {
                            'folder_name': folder_name,
                            'json_path': json_path,
                            'video_path': item['video_path']
                        }
            
            # No valid memory match found
            return False, None
        except Exception as e:
            return False, None

    def __getitem__(self, index):
        item = self.video_list[index % len(self.video_list)]
        folder_path = item['path']
        json_path = item['json_path']
        video_path = item['video_path']
        
        try:
            with open(json_path, 'r') as f:
                data_json = json.load(f)
            
            # Handle both dict and list formats
            if isinstance(data_json, list):
                # If JSON is a list, treat it as chunks directly
                chunks = data_json
                all_characters = {}  # No characters dict in list format
            elif isinstance(data_json, dict):
                # Standard format: dict with 'chunks' and 'characters'
                chunks = data_json.get('chunks', [])
                all_characters = data_json.get('characters', {})  # Dict of char_name -> list of descriptions
            else:
                raise ValueError(f"Unexpected JSON format in {json_path}: expected dict or list, got {type(data_json)}")
            
            # 1. Ignore last chunk
            if len(chunks) > 0:
                valid_chunks = chunks[:-1]
            else:
                valid_chunks = []
                
            if len(valid_chunks) == 0:
                # Case 1: Empty after removal
                raise ValueError(f"No valid chunks left after ignoring last one in: {folder_path}")

            use_memory = False
            extra_video = None
            
            # Selection Logic
            if len(valid_chunks) == 1:
                # Only one valid chunk
                core_chunk = valid_chunks[0]
                core_idx = 0
                use_memory = False
            else:
                # Multiple valid chunks
                # "选取除了第一个...以外的任意片段" -> Select from valid_chunks[1:]
                candidates_indices = list(range(1, len(valid_chunks)))
                core_idx = random.choice(candidates_indices)
                core_chunk = valid_chunks[core_idx]
                
                # Memory Selection Logic
                # "挑选memory时优先挑选核心片段的所有出现character都有出现的片段作为memory"
                # Identify Core Characters
                core_chars = core_chunk.get('character_list', [])
                
                # Check if core chunk has characters - if not, cannot use memory
                if not core_chars or len(core_chars) == 0:
                    use_memory = False
                else:
                    memory_candidates_indices = []
                    # Looking for memory in OTHER chunks? Or PREVIOUS chunks? 
                    # Assuming "memory" implies previous context, looking at indices 0 to core_idx-1
                    # But strict instruction: "挑选memory时...". If I look at ALL chunks except core?
                    # Usually memory is historical. I will restrict to previous valid chunks.
                    possible_memory_indices = list(range(0, core_idx))
                    
                    if len(possible_memory_indices) > 0:
                        # Strategy 1: All core characters appear
                        full_match_indices = []
                        max_match_count = -1
                        max_match_indices = []
                        
                        for m_idx in possible_memory_indices:
                            m_chunk = valid_chunks[m_idx]
                            m_chars_list = m_chunk.get('character_list', [])
                            
                            # Filter: Skip chunks with empty character_list
                            if not m_chars_list or len(m_chars_list) == 0:
                                continue
                            
                            m_chars = set(m_chars_list)
                            
                            match_count = 0
                            all_present = True
                            for c in core_chars:
                                if c in m_chars:
                                    match_count += 1
                                else:
                                    all_present = False
                            
                            if all_present:
                                full_match_indices.append(m_idx)
                            
                            if match_count > max_match_count:
                                max_match_count = match_count
                                max_match_indices = [m_idx]
                            elif match_count == max_match_count:
                                max_match_indices.append(m_idx)
                        
                        selected_memory_idx = -1
                        
                        if full_match_indices:
                            selected_memory_idx = random.choice(full_match_indices)
                        elif max_match_count > 0:
                            selected_memory_idx = random.choice(max_match_indices)
                        # else: max_match_count == 0 or empty -> Use memory False
                        
                        if selected_memory_idx != -1:
                            memory_chunk = valid_chunks[selected_memory_idx]
                            # Double-check: Ensure selected memory chunk has non-empty character_list
                            memory_chars = memory_chunk.get('character_list', [])
                            if memory_chars and len(memory_chars) > 0:
                                use_memory = True
                                # Load Extra Video
                                extra_video = self.load_video_segment(video_path, memory_chunk['start'], memory_chunk['end'], self.num_frames)
                            else:
                                use_memory = False
            
            # Build memory_text dictionary with character lists
            # Always include core_characters from core_chunk
            core_characters = core_chunk.get('character_list', [])
            
            # memory_characters only included if use_memory=True
            # Ensure memory_chunk exists and has non-empty character_list when use_memory=True
            if use_memory:
                if 'memory_chunk' not in locals():
                    # This should not happen, but safety check
                    use_memory = False
                    memory_characters = []
                else:
                    memory_characters = memory_chunk.get('character_list', [])
                    # Final safety check: if memory_characters is empty, set use_memory=False
                    if not memory_characters or len(memory_characters) == 0:
                        use_memory = False
                        memory_characters = []
                        extra_video = None  # Also clear extra_video if we're not using memory
            else:
                memory_characters = []
            
            memory_text = {
                'core_characters': core_characters,
                'memory_characters': memory_characters
            }

            # Load Core Video
            video = self.load_video_segment(video_path, core_chunk['start'], core_chunk['end'], self.num_frames)
            first_ref_frames, random_ref_frame = self.process_frames_for_ref(video)
            
            text = core_chunk['content']
            
            # Get folder name (minimum folder path)
            folder_name = os.path.basename(folder_path)
            
            # Get core_idx and memory_idx for cache naming
            core_idx_in_valid = core_idx  # Index in valid_chunks
            memory_idx_in_valid = selected_memory_idx if use_memory else -1
            if 'video_reader' in locals():
                video_reader.close()
            return {
                "text": text,
                "video": video,
                "path": video_path,
                "folder_name": folder_name,  # Minimum folder path for cache naming
                "core_idx": core_idx_in_valid,  # Index in valid_chunks
                "memory_idx": memory_idx_in_valid,  # Index in valid_chunks, -1 if no memory
                "first_ref_frames": first_ref_frames,
                "random_ref_frame": random_ref_frame,
                "use_memory": use_memory,
                "extra_video": extra_video,
                "memory_text": memory_text,
                "all_characters": all_characters  # For character descriptions lookup
            }

        except Exception as e:
            # print(f"Error loading {video_path}: {e}")
            raise e # User asked to error out

    def __len__(self):
        return len(self.video_list)


class CandidateGroupsDataset(Dataset):
    """
    Dataset from candidate_groups.csv: each row = (video_id, group_index, candidate_clips).
    Sample: random core from candidate_clips, random memory from that clip's overlapping_clip_indices (excluding core).
    Skips: (1) frame count mismatch, (2) empty overlapping after excluding core. Logs to skip_log_path.
    """
    def __init__(self, candidate_groups_csv, character_lists_dir, video_root, args, skip_log_path=None):
        self.args = args
        self.num_frames = getattr(args, 'num_frames', 81)
        self.height = getattr(args, 'height', 480)
        self.width = getattr(args, 'width', 832)
        self.candidate_groups_csv = os.path.abspath(candidate_groups_csv)
        self.character_lists_dir = os.path.abspath(character_lists_dir)
        self.video_root = os.path.abspath(video_root)
        self.skip_log_path = skip_log_path
        self._skip_log_file = None  # opened on first skip
        self.row_list = []  # list of (video_id, group_index, candidate_clips_list)
        self._build_row_list()
        self.frame_process = v2.Compose([
            v2.Resize(size=(self.height, self.width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _build_row_list(self):
        self.row_list = []
        with open(self.candidate_groups_csv, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                video_id = row.get('video_id', '').strip()
                try:
                    g = int(row.get('group_index', 0))
                except (ValueError, TypeError):
                    continue
                cand = row.get('candidate_clips', '')
                if not cand:
                    continue
                candidate_clips_list = [int(x.strip()) for x in cand.split('|') if x.strip().isdigit()]
                if not candidate_clips_list:
                    continue
                self.row_list.append((video_id, g, candidate_clips_list))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_skip_log_file'] = None
        state['row_list'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._skip_log_file = None
        if self.row_list is None:
            self._build_row_list()

    def _log_skip(self, video_id, group_index, clip_index, reason, extra=""):
        if not self.skip_log_path:
            return
        try:
            if self._skip_log_file is None:
                os.makedirs(os.path.dirname(self.skip_log_path) or ".", exist_ok=True)
                self._skip_log_file = open(self.skip_log_path, 'a', encoding='utf-8')
            self._skip_log_file.write(
                f"video_id={video_id}\tgroup_index={group_index}\tclip_index={clip_index}\treason={reason}\t{extra}\n"
            )
            self._skip_log_file.flush()
        except Exception:
            pass

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(
            image,
            (self.height, self.width),
        )
        return image

    def check_video_ok(self, video_path):
        """轻量检查：仅 ffprobe 查帧数，不解码。确认可加载后再解码，避免后知后觉。"""
        try:
            info = _ffmpeg_gpu_video_info(video_path)
            total_frames = info.get("num_frames") or 0
            return total_frames >= self.num_frames
        except Exception:
            return False

    def load_video_full(self, video_path):
        """Load full clip video and sample num_frames. 使用 FFmpeg GPU 解码。Returns tensor [C,T,H,W] or None if frame count too small."""
        try:
            info = _ffmpeg_gpu_video_info(video_path)
            total_frames = info["num_frames"]
            fps = info["fps"]
            duration = total_frames / fps if fps > 0 else total_frames / 24.0
            if total_frames < self.num_frames:
                return None
            return self.load_video_segment(video_path, 0.0, duration, self.num_frames)
        except RuntimeError:
            raise
        except Exception:
            return None

    def load_video_segment(self, video_path, start_time, end_time, target_num_frames=81):
        """使用 FFmpeg GPU (-hwaccel cuda) 解码。"""
        info = _ffmpeg_gpu_video_info(video_path)
        total_frames = info["num_frames"]
        fps = info["fps"]
        w, h = info["width"], info["height"]
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        end_frame = min(end_frame, total_frames)
        segment_len = end_frame - start_frame
        if segment_len < target_num_frames:
            return None
        stride = max(segment_len // target_num_frames, 1)
        frame_indices = []
        current_idx = start_frame
        while len(frame_indices) < target_num_frames:
            if current_idx >= end_frame:
                frame_indices.append(frame_indices[-1] if frame_indices else start_frame)
            else:
                frame_indices.append(current_idx)
                current_idx += stride
        safe_indices = [min(i, total_frames - 1) for i in frame_indices]
        batch = _ffmpeg_gpu_decode_frames(video_path, safe_indices, w, h, fps)
        frames = []
        for i in range(batch.shape[0]):
            frame = Image.fromarray(batch[i]).convert("RGB")
            frame = self.crop_and_resize(frame)
            frame = self.frame_process(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        return rearrange(frames, "T C H W -> C T H W")

    def process_frames_for_ref(self, video_tensor):
        C, T, H, W = video_tensor.shape
        num_ref = min(12, T)
        first_ref_frames = [video_tensor[:, i, :, :].clone() for i in range(num_ref)]
        random_idx = random.randint(0, T - 1)
        random_ref_frame = video_tensor[:, random_idx, :, :].clone()
        return first_ref_frames, random_ref_frame

    def __getitem__(self, index):
        index = index % len(self.row_list)
        video_id, group_index, candidate_clips_list = self.row_list[index]
        # 单次尝试，不重试
        core_clip = random.choice(candidate_clips_list)
        json_path = os.path.join(self.character_lists_dir, f"{video_id}.json")
        if not os.path.isfile(json_path):
            self._log_skip(video_id, group_index, core_clip, "no_json", "")
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            self._log_skip(video_id, group_index, core_clip, "json_error", "")
            return None
        shots = data.get('shots') or []
        shot = None
        core_clip_info = None
        for s in shots:
            if s.get('group_index') == group_index:
                shot = s
                break
        if shot is None:
            self._log_skip(video_id, group_index, core_clip, "group_not_found", "")
            return None
        for c in shot.get('clips') or []:
            if c.get('clip_index') == core_clip:
                core_clip_info = c
                break
        if core_clip_info is None:
            self._log_skip(video_id, group_index, core_clip, "clip_not_found", "")
            return None
        overlapping = core_clip_info.get('overlapping_clip_indices') or []
        memory_candidates = [x for x in overlapping if x != core_clip]
        if not memory_candidates:
            self._log_skip(video_id, group_index, core_clip, "empty_overlapping", "")
            return None
        memory_clip = random.choice(memory_candidates)
        group_dir = os.path.join(self.video_root, video_id, f"group_{group_index}")
        core_path = os.path.join(group_dir, f"clip{core_clip}.mp4")
        memory_path = os.path.join(group_dir, f"clip{memory_clip}.mp4")
        if not os.path.isfile(core_path) or not os.path.isfile(memory_path):
            self._log_skip(video_id, group_index, core_clip, "video_missing", f"core={core_path} mem={memory_path}")
            return None
        # 先轻量检查 core 和 memory 帧数，通过后再解码，避免后知后觉
        if not self.check_video_ok(core_path):
            self._log_skip(video_id, group_index, core_clip, "frame_count_core", "")
            return None
        if not self.check_video_ok(memory_path):
            self._log_skip(video_id, group_index, memory_clip, "frame_count_memory", "")
            return None
        video = self.load_video_full(core_path)
        if video is None:
            self._log_skip(video_id, group_index, core_clip, "frame_count_core", "")
            return None
        extra_video = self.load_video_full(memory_path)
        if extra_video is None:
            self._log_skip(video_id, group_index, memory_clip, "frame_count_memory", "")
            return None
        first_ref_frames, random_ref_frame = self.process_frames_for_ref(video)
        core_characters = core_clip_info.get('characters') or []
        memory_clip_info = None
        for c in shot.get('clips') or []:
            if c.get('clip_index') == memory_clip:
                memory_clip_info = c
                break
        memory_characters = (memory_clip_info.get('characters') or []) if memory_clip_info else []
        memory_caption = memory_clip_info.get('caption', '') if memory_clip_info else ''
        text = core_clip_info.get('caption', '')
        folder_name = f"{video_id}_group{group_index}_c{core_clip}_m{memory_clip}"
        memory_text = {
            'core_characters': core_characters,
            'memory_characters': memory_characters,
            'prompt': memory_caption,
        }
        return {
            "text": text,
            "video": video,
            "path": core_path,
            "folder_name": folder_name,
            "first_ref_frames": first_ref_frames,
            "random_ref_frame": random_ref_frame,
            "use_memory": True,
            "extra_video": extra_video,
            "memory_text": memory_text,
            "video_id": video_id,
            "group_index": group_index,
            "core_clip_index": core_clip,
            "memory_clip_index": memory_clip,
        }

    def __len__(self):
        return len(self.row_list)


def test():
    args = MockArgs()
    dataset = TextVideoDataset_New(args.dataset_path, args)
    
    print("Testing iteration...")
    # Test random 50 items
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    success = 0
    errors = 0
    
    for i in tqdm(indices[:50]):
        try:
            data = dataset[i]
            # print(f"Success: {data['path']} - Memory: {data['use_memory']}")
            # Basic validation
            assert data['video'].shape == (3, 81, 480, 832)
            if data['use_memory']:
                assert data['extra_video'] is not None
                assert data['extra_video'].shape == (3, 81, 480, 832)
            else:
                assert data['extra_video'] is None
            
            # Check memory text structure (should be a dict with 'core_characters' and 'memory_characters')
            assert isinstance(data['memory_text'], dict), f"memory_text should be a dict, got {type(data['memory_text'])}"
            assert 'core_characters' in data['memory_text'], "memory_text should contain 'core_characters'"
            assert 'memory_characters' in data['memory_text'], "memory_text should contain 'memory_characters'"
            assert isinstance(data['memory_text']['core_characters'], list), "core_characters should be a list"
            assert isinstance(data['memory_text']['memory_characters'], list), "memory_characters should be a list"
            
            # If use_memory=True, memory_characters should not be empty; if False, should be empty
            if data['use_memory']:
                assert len(data['memory_text']['memory_characters']) > 0, "memory_characters should not be empty when use_memory=True"
            else:
                assert len(data['memory_text']['memory_characters']) == 0, "memory_characters should be empty when use_memory=False"
            
            # print(f"Core characters: {data['memory_text']['core_characters']}")
            # print(f"Memory characters: {data['memory_text']['memory_characters']}")
                
            success += 1
        except Exception as e:
            print(f"FAILED: {dataset.video_list[i]['path']} - {e}")
            errors += 1
            # break # Uncomment to stop on first error for debugging

    print(f"Test Complete. Success: {success}, Errors: {errors}")

if __name__ == "__main__":
    test()
