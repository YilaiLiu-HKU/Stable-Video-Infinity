import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffsynth.core.loader import ModelConfig
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video_svi_pro import WanVideoSviProPipeline


def _sorted_glob(pattern, required=True):
    paths = sorted(glob.glob(pattern))
    if required and len(paths) <= 0:
        raise FileNotFoundError(f"Missing required checkpoint files: {pattern}")
    return paths


def _get_module_dtype_device(module, default_device="cpu", default_dtype=torch.bfloat16):
    if module is None:
        return default_dtype, torch.device(default_device)
    if hasattr(module, "dtype") and hasattr(module, "device"):
        dtype = getattr(module, "dtype", default_dtype)
        device = getattr(module, "device", default_device)
        return dtype, torch.device(device)
    inner_model = getattr(module, "model", None)
    if inner_model is not None:
        try:
            param = next(inner_model.parameters())
            return param.dtype, param.device
        except Exception:
            pass
    try:
        param = next(module.parameters())
        return param.dtype, param.device
    except Exception:
        return default_dtype, torch.device(default_device)


class WanTrainingSchedulerAdapter:
    def __init__(self, scheduler):
        self._scheduler = scheduler

    def __getattr__(self, name):
        return getattr(self._scheduler, name)

    def set_timesteps(self, num_inference_steps, shift=1.0):
        self._scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            training=True,
            shift=shift,
        )
        return self._scheduler.timesteps

    def get_timesteps(self, num_inference_steps, denoising_strength=1.0, shift=5.0):
        self._scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            shift=shift,
        )
        return self._scheduler.timesteps

    def training_target(self, latents, noise, timestep):
        return self._scheduler.training_target(latents, noise, timestep)

    def training_weight(self, timestep):
        return self._scheduler.training_weight(timestep)


class WanPrompterAdapter:
    def __init__(self, pipe):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

    def encode_prompt(self, prompt, positive=True, device="cuda"):
        del positive
        target_device = torch.device(device)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(target_device)
        mask = mask.to(target_device)
        self.text_encoder.to(target_device)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb


class _DiffSynthSelfAttentionAdapter(nn.Module):
    def __init__(self, base_attn, freq_builder):
        super().__init__()
        self.base_attn = base_attn
        self._freq_builder = freq_builder

    def forward(self, x, seq_lens=None, grid_sizes=None, freqs=None):
        del seq_lens, freqs
        expanded_freqs = self._freq_builder(grid_sizes, x.device)
        return self.base_attn(x, expanded_freqs)


class _DiffSynthCrossAttentionAdapter(nn.Module):
    def __init__(self, base_attn):
        super().__init__()
        self.base_attn = base_attn

    def forward(self, x, context, context_lens=None):
        del context_lens
        return self.base_attn(x, context)


class _DiffSynthBlockAdapter(nn.Module):
    def __init__(self, base_block, freq_builder):
        super().__init__()
        self.base_block = base_block
        self.self_attn = _DiffSynthSelfAttentionAdapter(base_block.self_attn, freq_builder)
        self.cross_attn = _DiffSynthCrossAttentionAdapter(base_block.cross_attn)
        self.norm1 = base_block.norm1
        self.norm2 = base_block.norm2
        self.norm3 = base_block.norm3
        self.ffn = base_block.ffn
        self.modulation = base_block.modulation

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        del seq_lens, freqs, context_lens
        expanded_freqs = self.self_attn._freq_builder(grid_sizes, x.device)
        return self.base_block(x, context, e, expanded_freqs)


class WanDiffSynthVideoAdapter(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.dim = base_model.dim
        self.freq_dim = base_model.freq_dim
        self.patch_size = tuple(base_model.patch_size)
        self.patch_embedding = base_model.patch_embedding
        self.text_embedding = base_model.text_embedding
        self.time_embedding = base_model.time_embedding
        self.time_projection = base_model.time_projection
        self.head = base_model.head
        self.has_image_input = bool(getattr(base_model, "has_image_input", False))
        self.require_vae_embedding = bool(getattr(base_model, "require_vae_embedding", True))
        self.require_clip_embedding = bool(getattr(base_model, "require_clip_embedding", True))
        self.has_image_pos_emb = bool(getattr(base_model, "has_image_pos_emb", False))
        self.img_emb = getattr(base_model, "img_emb", None)
        self._freqs_tuple = base_model.freqs
        # Keep a tensor-like attr for existing training code paths that expect
        # `dit_model.freqs.device` to exist.
        self.freqs = base_model.freqs[0]
        self.blocks = nn.ModuleList(
            [_DiffSynthBlockAdapter(block, self._expand_freqs) for block in base_model.blocks]
        )

    def _expand_freqs(self, grid_sizes, device):
        if isinstance(grid_sizes, torch.Tensor):
            if grid_sizes.ndim == 2:
                f, h, w = [int(v) for v in grid_sizes[0].tolist()]
            else:
                f, h, w = [int(v) for v in grid_sizes.tolist()]
        else:
            f, h, w = [int(v) for v in grid_sizes]
        return torch.cat(
            [
                self._freqs_tuple[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self._freqs_tuple[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self._freqs_tuple[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(device=device)

    def patchify(self, x):
        x = self.base_model.patchify(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x, grid_size):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x,
        timestep,
        context,
        clip_feature=None,
        y=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        del kwargs
        model_dtype = self.patch_embedding.weight.dtype
        x = x.to(dtype=model_dtype)
        context = context.to(device=x.device, dtype=model_dtype)
        if y is not None:
            y = y.to(device=x.device, dtype=model_dtype)
            x = torch.cat([x, y], dim=1)

        x, (f, h, w) = self.patchify(x)
        b, seq_len, _ = x.shape
        device = x.device

        grid_sizes = torch.tensor([[f, h, w]] * b, device=device, dtype=torch.long)
        seq_lens = torch.full((b,), seq_len, device=device, dtype=torch.long)

        t_input = timestep.to(device=device)
        if t_input.dim() == 1:
            t_input = t_input.expand(t_input.size(0), seq_len)
        bt = t_input.size(0)
        t_flat = t_input.flatten()

        with torch.amp.autocast("cuda", dtype=torch.float32):
            t_embed = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t_flat)
                .unflatten(0, (bt, seq_len))
                .float()
                .to(device=device)
            )
            t_seq = self.time_projection(t_embed).unflatten(2, (6, self.dim))

        with torch.amp.autocast("cuda", dtype=model_dtype):
            context_emb = self.text_embedding(context)
            if (
                self.has_image_input
                and clip_feature is not None
                and self.img_emb is not None
                and self.require_clip_embedding
            ):
                clip_feature = clip_feature.to(device=device, dtype=model_dtype)
                clip_emb = self.img_emb(clip_feature)
                context_emb = torch.cat([clip_emb, context_emb], dim=1)

            for block in self.blocks:
                if self.training and use_gradient_checkpointing:
                    def _custom_forward(x_in, e_in):
                        return block(
                            x_in,
                            e=e_in,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context_emb,
                            context_lens=None,
                        )

                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                _custom_forward,
                                x,
                                t_seq,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            _custom_forward,
                            x,
                            t_seq,
                            use_reentrant=False,
                        )
                else:
                    x = block(
                        x,
                        e=t_seq,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context_emb,
                        context_lens=None,
                    )

            x = self.head(x, t_embed.float())
        return self.unpatchify(x, (f, h, w))


class WanTrainPipelineAdapter:
    def __init__(
        self,
        ckpt_dir,
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        task="i2v-A14B",
        train_noise_domain="low_noise",
    ):
        del task
        self.ckpt_dir = ckpt_dir
        self.device = str(device)
        self.torch_dtype = torch_dtype
        self.train_noise_domain = str(train_noise_domain or "low_noise").strip().lower()
        if self.train_noise_domain not in ("low_noise", "high_noise"):
            raise ValueError(
                f"train_noise_domain must be 'low_noise' or 'high_noise', got {self.train_noise_domain!r}"
            )

        high_noise_paths = _sorted_glob(
            os.path.join(ckpt_dir, "high_noise_model", "diffusion_pytorch_model*.safetensors")
        )
        low_noise_paths = _sorted_glob(
            os.path.join(ckpt_dir, "low_noise_model", "diffusion_pytorch_model*.safetensors")
        )
        text_encoder_path = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        vae_path = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
        tokenizer_path = os.path.join(ckpt_dir, "google", "umt5-xxl")
        image_encoder_path = os.path.join(
            ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        )

        model_configs = [
            ModelConfig(path=high_noise_paths, offload_device="cpu"),
            ModelConfig(path=low_noise_paths, offload_device="cpu"),
            ModelConfig(path=text_encoder_path, offload_device="cpu"),
            ModelConfig(path=vae_path, offload_device="cpu"),
        ]
        if os.path.exists(image_encoder_path):
            model_configs.append(ModelConfig(path=image_encoder_path, offload_device="cpu"))

        raw_pipe = WanVideoSviProPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(path=tokenizer_path, skip_download=True),
            redirect_common_files=False,
        )

        self._base_pipe = raw_pipe
        self.scheduler = WanTrainingSchedulerAdapter(raw_pipe.scheduler)
        self.prompter = WanPrompterAdapter(raw_pipe)
        self.vae = raw_pipe.vae
        self.image_encoder = raw_pipe.image_encoder
        self.high_noise_model = WanDiffSynthVideoAdapter(raw_pipe.dit) if raw_pipe.dit is not None else None
        self.low_noise_model = WanDiffSynthVideoAdapter(raw_pipe.dit2) if raw_pipe.dit2 is not None else None
        self.dit = None
        self.set_active_noise_domain(self.train_noise_domain)
        self.training = True

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._base_pipe, name)

    def set_active_noise_domain(self, noise_domain):
        domain = str(noise_domain or self.train_noise_domain).strip().lower()
        if domain not in ("low_noise", "high_noise"):
            domain = self.train_noise_domain
        self.dit = self.high_noise_model if domain == "high_noise" else self.low_noise_model
        if self.dit is None:
            self.dit = self.high_noise_model if self.high_noise_model is not None else self.low_noise_model
        return self.dit

    def set_active_noise_domain_from_timestep(self, timestep, boundary_ratio=0.9):
        del timestep, boundary_ratio
        return self.set_active_noise_domain(self.train_noise_domain)

    def denoising_model(self):
        return self.dit

    def preprocess_image(self, image, **kwargs):
        return self._base_pipe.preprocess_image(
            image,
            torch_dtype=self.torch_dtype,
            device=self.device,
            **kwargs,
        )

    def prepare_extra_input(self, latents=None):
        del latents
        return {}

    def encode_prompt(self, prompt, positive=True):
        return self.prompter.encode_prompt(prompt, positive=positive, device=self.device)

    def encode_images_adaptive(
        self,
        first_frames,
        random_ref_frame,
        num_frames,
        height,
        width,
        use_first_aug=False,
        ref_pad_cfg=False,
        ref_pad_num=None,
    ):
        del random_ref_frame, use_first_aug, ref_pad_cfg, ref_pad_num
        if not first_frames:
            raise ValueError("encode_images_adaptive requires at least one frame")

        active_dit = self.denoising_model()
        out = {}
        vae_dtype, vae_device = _get_module_dtype_device(
            self.vae,
            default_device=self.device,
            default_dtype=self.torch_dtype,
        )
        image_dtype, image_device = _get_module_dtype_device(
            self.image_encoder,
            default_device=self.device,
            default_dtype=self.torch_dtype,
        )
        if (
            self.image_encoder is not None
            and active_dit is not None
            and bool(getattr(active_dit, "require_clip_embedding", True))
        ):
            image = self.preprocess_image(first_frames[0].resize((width, height))).to(
                device=image_device,
                dtype=image_dtype,
            )
            clip_context = self.image_encoder.encode_image([image])
            out["clip_feature"] = clip_context.to(dtype=self.torch_dtype, device=self.device)

        if active_dit is None or not bool(getattr(active_dit, "require_vae_embedding", True)):
            return out

        anchor = self.preprocess_image(first_frames[0].resize((width, height))).to(
            device=vae_device,
            dtype=vae_dtype,
        )
        anchor_latent = self.vae.encode(
            [anchor.transpose(0, 1).to(dtype=vae_dtype, device=vae_device)],
            device=vae_device,
            tiled=False,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )[0].to(device=self.device, dtype=self.torch_dtype)
        total_latents = (int(num_frames) - 1) // 4 + 1
        padding_size = total_latents - int(anchor_latent.shape[1])
        padding = torch.zeros(
            anchor_latent.shape[0],
            padding_size,
            anchor_latent.shape[2],
            anchor_latent.shape[3],
            dtype=self.torch_dtype,
            device=self.device,
        )
        y_latent = torch.cat([anchor_latent, padding], dim=1)

        msk = torch.ones(1, int(num_frames), height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8).transpose(1, 2)[0]
        out["y"] = torch.cat([msk, y_latent], dim=0).unsqueeze(0).to(
            dtype=self.torch_dtype,
            device=self.device,
        )
        return out


def build_wan22_training_pipe(
    ckpt_dir,
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    task="i2v-A14B",
    train_noise_domain="low_noise",
):
    return WanTrainPipelineAdapter(
        ckpt_dir=ckpt_dir,
        device=device,
        torch_dtype=torch_dtype,
        task=task,
        train_noise_domain=train_noise_domain,
    )
