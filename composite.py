"""Pixel-space compositing -- the hard "only skin changes" guarantee.

Every stage of the pipeline ends here. Nothing downstream is allowed to write a
full-canvas VAE decode over the input: the output is always assembled as
``base * (1 - m) + candidate * m`` in float32 pixel space, and pixels where the
mask is exactly zero are copied straight from the input so they stay
bit-identical.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import comfy.utils


def gaussian_kernel1d(sigma: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalised 1-D gaussian kernel spanning +/- 3 sigma."""
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _blur_bchw(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur of a ``[B, C, H, W]`` tensor."""
    kernel = gaussian_kernel1d(sigma, image.dtype).to(image.device)
    pad = (kernel.numel() - 1) // 2
    channels = image.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    out = F.pad(image, (pad, pad, 0, 0), mode="replicate")
    out = F.conv2d(out, horizontal, groups=channels)
    out = F.pad(out, (0, 0, pad, pad), mode="replicate")
    return F.conv2d(out, vertical, groups=channels)


def gaussian_blur_image(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Low-pass a ``[B, H, W, C]`` image.

    Wide kernels are approximated by blurring at reduced resolution, which keeps
    a 2 MP low-pass in the tens of milliseconds instead of seconds. The result
    only ever feeds a low-frequency difference, so the approximation is
    invisible.
    """
    if sigma <= 0.0:
        return image
    work = image.movedim(-1, 1)
    factor = int(min(8, max(1, round(sigma / 2.0)))) if sigma > 3.0 else 1
    if factor > 1:
        height, width = work.shape[-2], work.shape[-1]
        small = F.interpolate(
            work,
            size=(max(1, height // factor), max(1, width // factor)),
            mode="area",
        )
        small = _blur_bchw(small, sigma / factor)
        work = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    else:
        work = _blur_bchw(work, sigma)
    return work.movedim(1, -1)


def lock_low_frequency(
    base: torch.Tensor,
    candidate: torch.Tensor,
    amount: float,
    sigma: float | None = None,
) -> torch.Tensor:
    """Keep ``base``'s colour and shading, import only ``candidate``'s fine texture.

    Suppresses broad colour/shading changes. This is a frequency filter, not a
    semantic safeguard: small invented marks and edges can still pass through.

    Args:
        base: ``[B, H, W, C]`` original image.
        candidate: ``[B, H, W, C]`` processed image.
        amount: 0 = leave the candidate alone, 1 = fully lock the low band.
        sigma: low-pass radius; defaults to a resolution-relative value.

    Returns:
        ``[B, H, W, C]`` corrected candidate, clamped to 0..1.

    The default cutoff scales with image resolution; attenuation depends on
    structure size and does not perfectly separate pores from other marks.
    """
    if amount <= 0.0:
        return candidate
    if sigma is None:
        sigma = max(1.5, min(base.shape[1], base.shape[2]) / 400.0)
    base = base.to(torch.float32)
    candidate = candidate.to(device=base.device, dtype=torch.float32)
    candidate = torch.where(torch.isfinite(candidate), candidate, base)
    residual = candidate - base
    corrected = candidate - gaussian_blur_image(residual, sigma) * float(min(1.0, amount))
    return corrected.clamp(0.0, 1.0)


def _match_mask(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``[B, H, W]`` (or ``[H, W]``) mask onto a ``[B, H, W, C]`` image."""
    mask = mask.to(device=image.device, dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
    if mask.shape[-2:] != image.shape[1:3]:
        mask = comfy.utils.common_upscale(
            mask.unsqueeze(1), image.shape[2], image.shape[1], "bilinear", crop="disabled"
        ).squeeze(1)
    if mask.shape[0] != image.shape[0]:
        mask = comfy.utils.repeat_to_batch_size(mask, image.shape[0])
    return mask.unsqueeze(-1).clamp(0.0, 1.0)


def composite_masked(
    base: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Blend ``candidate`` into ``base`` through ``mask``, in float32 pixel space.

    Args:
        base: ``[B, H, W, C]`` untouched source image.
        candidate: ``[B, H, W, C]`` processed image, same shape as ``base``.
        mask: ``[B, H, W]`` blend weights in 0..1.

    Returns:
        ``[B, H, W, C]`` image whose ``mask == 0`` pixels are bit-identical to
        ``base``. NaN/Inf produced upstream cannot leak into untouched regions
        because the zero-mask branch is selected, not multiplied.
    """
    base32 = base.to(torch.float32)
    cand32 = candidate.to(device=base32.device, dtype=torch.float32)
    if cand32.shape != base32.shape:
        cand32 = comfy.utils.common_upscale(
            cand32.movedim(-1, 1), base32.shape[2], base32.shape[1], "bilinear", crop="disabled"
        ).movedim(1, -1)

    weights = _match_mask(mask, base32)
    cand32 = torch.where(torch.isfinite(cand32), cand32, base32).clamp(0.0, 1.0)
    blended = base32 + (cand32 - base32) * weights
    return torch.where(weights > 0.0, blended, base32)


def blend_scaled(
    base: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor, strength: float
) -> torch.Tensor:
    """``composite_masked`` with the mask scaled by a 0..1 strength."""
    if strength <= 0.0:
        return base.to(torch.float32)
    return composite_masked(base, candidate, mask * float(min(1.0, strength)))


def align_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    pad: int,
    stride: int,
    min_size: int = 0,
) -> tuple[int, int, int, int]:
    """Pad a bounding box, snap it to ``stride``, and clamp it to the canvas.

    The snapped box is guaranteed to have both sides divisible by ``stride``
    whenever the canvas itself is large enough, so VAE encode/decode round-trips
    return the exact crop size.
    """
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    if min_size > 0:
        if x2 - x1 < min_size:
            grow = min_size - (x2 - x1)
            x1 = max(0, x1 - grow // 2)
            x2 = min(width, x1 + min_size)
            x1 = max(0, x2 - min_size)
        if y2 - y1 < min_size:
            grow = min_size - (y2 - y1)
            y1 = max(0, y1 - grow // 2)
            y2 = min(height, y1 + min_size)
            y1 = max(0, y2 - min_size)

    if stride > 1:
        def _round_up(value: int) -> int:
            return ((value + stride - 1) // stride) * stride

        # Round the size up to a stride multiple, but never past what the canvas
        # can hold as a stride multiple.
        box_w = min(_round_up(x2 - x1), max(stride, (width // stride) * stride))
        box_h = min(_round_up(y2 - y1), max(stride, (height // stride) * stride))
        box_w = min(box_w, width)
        box_h = min(box_h, height)
        # Shift the origin so the enlarged box still fits inside the canvas.
        x1 = max(0, min(x1, width - box_w))
        y1 = max(0, min(y1, height - box_h))
        x2, y2 = x1 + box_w, y1 + box_h

    return int(x1), int(y1), int(max(x2, x1 + 1)), int(max(y2, y1 + 1))


def crop_image(image: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    """Crop ``[B, H, W, C]`` with an ``(x1, y1, x2, y2)`` box."""
    x1, y1, x2, y2 = box
    return image[:, y1:y2, x1:x2, :]


def merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Union overlapping rectangles, including overlaps introduced by alignment."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                b = boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3]))
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    return boxes


def crop_mask(mask: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    """Crop ``[B, H, W]`` with an ``(x1, y1, x2, y2)`` box."""
    x1, y1, x2, y2 = box
    return mask[:, y1:y2, x1:x2]


def paste_image(
    canvas: torch.Tensor, patch: torch.Tensor, box: tuple[int, int, int, int]
) -> torch.Tensor:
    """Write ``patch`` into a copy of ``canvas`` at ``box`` (no blending)."""
    x1, y1, x2, y2 = box
    out = canvas.clone()
    if patch.shape[1] != (y2 - y1) or patch.shape[2] != (x2 - x1):
        patch = comfy.utils.common_upscale(
            patch.movedim(-1, 1), x2 - x1, y2 - y1, "bicubic", crop="disabled"
        ).movedim(1, -1)
    out[:, y1:y2, x1:x2, :] = patch.to(dtype=out.dtype, device=out.device)
    return out
