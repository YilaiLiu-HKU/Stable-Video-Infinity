from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


DEFAULT_YOLO_SEG_CKPT = "/data/yilai/yolo26x-seg.pt"


@dataclass
class _TensorFrameMeta:
    had_batch_dim: bool
    layout: str  # "hwc" or "chw"


_YOLO_CACHE: dict[str, Any] = {}


def _get_yolo_cpu_model(model_path: str | Path) -> Any:
    if YOLO is None:
        raise RuntimeError("ultralytics is unavailable; cannot run CPU people pixelation")

    model_path = str(Path(model_path).expanduser().resolve())
    if model_path in _YOLO_CACHE:
        return _YOLO_CACHE[model_path]

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"YOLO ckpt not found for CPU pixelation: {p}")

    model = YOLO(model_path)
    _YOLO_CACHE[model_path] = model
    return model


def _tensor_to_hwc_u8(frame: torch.Tensor) -> tuple[np.ndarray, _TensorFrameMeta]:
    t = frame.detach().cpu()
    had_batch_dim = False
    if t.dim() == 4 and int(t.shape[0]) == 1:
        t = t[0]
        had_batch_dim = True

    if t.dim() != 3:
        raise ValueError(f"Expected frame tensor with 3 dims (or 1x3 dims), got shape={tuple(frame.shape)}")

    if int(t.shape[-1]) == 3:
        layout = "hwc"
        hwc = t
    elif int(t.shape[0]) == 3:
        layout = "chw"
        hwc = t.permute(1, 2, 0)
    else:
        raise ValueError(f"Unsupported frame layout for shape={tuple(frame.shape)}")

    if hwc.dtype != torch.uint8:
        hwc = hwc.to(torch.uint8)

    arr = hwc.numpy()
    return arr, _TensorFrameMeta(had_batch_dim=had_batch_dim, layout=layout)


def _hwc_u8_to_tensor(arr_hwc: np.ndarray, meta: _TensorFrameMeta) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr_hwc)).to(torch.uint8)
    if meta.layout == "chw":
        t = t.permute(2, 0, 1)
    if meta.had_batch_dim:
        t = t.unsqueeze(0)
    return t


def _build_people_mask_from_bgr(
    image_bgr: np.ndarray,
    model: Any,
    conf: float,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    results = model(image_bgr, conf=float(conf), verbose=False, device="cpu")
    if not results:
        return full_mask

    result = results[0]
    if result.masks is None or result.boxes is None:
        return full_mask

    names = result.names
    person_ids = {int(cls_id) for cls_id, cls_name in names.items() if str(cls_name) == "person"}
    if not person_ids:
        return full_mask

    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    for i, cls_id in enumerate(classes):
        if int(cls_id) not in person_ids:
            continue
        contours = result.masks.xy[i]
        if contours is None or len(contours) == 0:
            continue
        contour = contours.astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(full_mask, [contour], -1, color=255, thickness=cv2.FILLED)

    return full_mask


def _pixelate_masked_region_bgr(image_bgr: np.ndarray, mask: np.ndarray, pixel_block_size: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    block = max(2, int(pixel_block_size))
    small_w = max(1, w // block)
    small_h = max(1, h // block)

    small = cv2.resize(image_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_bool = mask > 0
    out = image_bgr.copy()
    out[mask_bool] = pixelated[mask_bool]
    return out


def pixelate_people_tensor_cpu(
    frame_tensor: torch.Tensor,
    model: Any,
    conf: float = 0.25,
    pixel_block_size: int = 12,
    mask_dilate_kernel: int = 9,
) -> tuple[torch.Tensor, bool]:
    """Pixelate people in a single frame tensor on CPU and keep input tensor layout."""
    frame_hwc_u8, meta = _tensor_to_hwc_u8(frame_tensor)
    frame_bgr = cv2.cvtColor(frame_hwc_u8, cv2.COLOR_RGB2BGR)

    mask = _build_people_mask_from_bgr(frame_bgr, model=model, conf=conf)
    if mask_dilate_kernel > 1:
        kernel = np.ones((int(mask_dilate_kernel), int(mask_dilate_kernel)), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    if int(mask.max()) <= 0:
        return frame_tensor.detach().cpu().to(torch.uint8), False

    pixelated_bgr = _pixelate_masked_region_bgr(frame_bgr, mask, pixel_block_size=pixel_block_size)
    pixelated_rgb = cv2.cvtColor(pixelated_bgr, cv2.COLOR_BGR2RGB)
    out_tensor = _hwc_u8_to_tensor(pixelated_rgb, meta)
    return out_tensor, True


def maybe_pixelate_condition_batch_cpu(
    batch: dict,
    enabled: bool,
    model_path: str | Path = DEFAULT_YOLO_SEG_CKPT,
    conf: float = 0.25,
    pixel_block_size: int = 12,
    mask_dilate_kernel: int = 9,
) -> tuple[dict, bool]:
    """Apply CPU people-pixelation on first_ref_frames/random_ref_frame in a training batch."""
    if not enabled:
        return batch, False

    new_batch = dict(batch)
    model = _get_yolo_cpu_model(model_path)
    changed = False

    raw_frf = batch.get("first_ref_frames")
    if isinstance(raw_frf, list):
        processed_frames = []
        for frame in raw_frf:
            if isinstance(frame, torch.Tensor):
                out_frame, frame_changed = pixelate_people_tensor_cpu(
                    frame,
                    model=model,
                    conf=conf,
                    pixel_block_size=pixel_block_size,
                    mask_dilate_kernel=mask_dilate_kernel,
                )
                processed_frames.append(out_frame)
                changed = changed or bool(frame_changed)
            else:
                processed_frames.append(frame)
        new_batch["first_ref_frames"] = processed_frames
    elif isinstance(raw_frf, torch.Tensor):
        out_frf, frame_changed = pixelate_people_tensor_cpu(
            raw_frf,
            model=model,
            conf=conf,
            pixel_block_size=pixel_block_size,
            mask_dilate_kernel=mask_dilate_kernel,
        )
        new_batch["first_ref_frames"] = out_frf
        changed = changed or bool(frame_changed)

    raw_rrf = batch.get("random_ref_frame")
    if isinstance(raw_rrf, torch.Tensor):
        out_rrf, frame_changed = pixelate_people_tensor_cpu(
            raw_rrf,
            model=model,
            conf=conf,
            pixel_block_size=pixel_block_size,
            mask_dilate_kernel=mask_dilate_kernel,
        )
        new_batch["random_ref_frame"] = out_rrf
        changed = changed or bool(frame_changed)

    if changed:
        # Any frame mutation invalidates extraction-side precomputed image embedding.
        new_batch["precomputed_image_emb"] = None

    return new_batch, changed
