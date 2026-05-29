#!/usr/bin/env python3
"""
Nearest-neighbor patch retrieval visualization for PatchCore.

For each of the top-K most anomalous patches in a test image, this script:
  1. Shows a PCA projection of the feature space — the memory bank (normal
     training patches) as a gray cloud, all test patches colored by their
     anomaly score, and the top-K anomalous patches with their nearest
     memory-bank neighbours connected by dashed lines.
  2. Shows pixel-level crops of those patches alongside their nearest normal
     match from the training set.

Usage:
  python visualize_nn.py \\
    --checkpoint results/bottle/Patchcore/.../model.ckpt \\
    --image path/to/test.png \\
    --train-dir data/mvtec_anomaly_detection/bottle/train/good \\
    --top-k 3

  python visualize_nn.py ... --output my_viz.png --show
"""

import argparse
from pathlib import Path

import anomalib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.models import Patchcore
from PIL import Image

torch.serialization.add_safe_globals([anomalib.PrecisionType])

_MEAN   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD    = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


# ── image helpers ────────────────────────────────────────────────────────────

def _load(path: Path, size: tuple[int, int]) -> tuple[torch.Tensor, np.ndarray]:
    """Return (model_tensor (1,3,H,W), rgb as float32 (H,W,3) in [0,1])."""
    img = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb).permute(2, 0, 1)
    t   = (t - _MEAN) / _STD
    return t.unsqueeze(0), rgb


def _crop(rgb: np.ndarray, row: int, col: int,
          feat_h: int, feat_w: int, context_px: int) -> np.ndarray:
    """Crop a square pixel region centred on feature-grid position (row, col)."""
    h, w = rgb.shape[:2]
    cy   = int((row + 0.5) * h / feat_h)
    cx   = int((col + 0.5) * w / feat_w)
    half = context_px // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    crop = rgb[y0:y1, x0:x1]
    out  = np.zeros((context_px, context_px, 3), dtype=crop.dtype)
    out[:crop.shape[0], :crop.shape[1]] = crop
    return out


# ── feature helpers ──────────────────────────────────────────────────────────

def _model_device(m) -> torch.device:
    try:
        return next(m.feature_extractor.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _extract_patches(
    m,
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int]:
    """Return (patches (N, D) on model device, feat_h, feat_w)."""
    device = _model_device(m)
    tensor = tensor.to(device=device, dtype=dtype)
    with torch.no_grad():
        feats  = m.feature_extractor(tensor)
        feats  = {layer: m.feature_pooler(f) for layer, f in feats.items()}
        emb    = m.generate_embedding(feats)       # (1, D, feat_h, feat_w)
        _, _, feat_h, feat_w = emb.shape
        patches = m.reshape_embedding(emb)         # (feat_h * feat_w, D)
    return patches, feat_h, feat_w


def _build_source_index(
    m,
    train_dir: Path,
    query_mb_indices: list[int],
    img_size: tuple[int, int],
    dtype: torch.dtype,
) -> dict[int, tuple[Path, int, int, int, int]]:
    """For each memory-bank index, find the training patch with minimum
    Euclidean distance to it.

    Returns: mb_index → (src_img_path, patch_row, patch_col, feat_h, feat_w)

    Because the coreset is an exact subset of training embeddings, the
    minimum distance here is essentially zero.
    """
    query_vecs = m.memory_bank[query_mb_indices]   # (K, D) on model device
    K          = len(query_mb_indices)
    best_dist  = torch.full((K,), float("inf"))
    best: dict[int, tuple[Path, int, int, int, int]] = {}

    img_paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")
                       for p in train_dir.glob(ext))
    print(f"  Scanning {len(img_paths)} training images …")

    for img_path in img_paths:
        tensor, _ = _load(img_path, img_size)
        patches, feat_h, feat_w = _extract_patches(m, tensor, dtype)
        dists = m.euclidean_dist(patches, query_vecs).cpu()   # (N_patches, K)

        for k, mb_idx in enumerate(query_mb_indices):
            min_dist, min_flat = dists[:, k].min(0)
            if min_dist.item() < best_dist[k].item():
                best_dist[k] = min_dist
                pr = int(min_flat) // feat_w
                pc = int(min_flat) % feat_w
                best[mb_idx] = (img_path, pr, pc, feat_h, feat_w)

    return best




# ── main visualisation ───────────────────────────────────────────────────────

def visualize(
    checkpoint: str | Path,
    image:      str | Path,
    train_dir:  str | Path,
    top_k:      int = 5,
    context_px: int = 48,
    output:     str | Path | None = None,
    show:       bool = False,
) -> None:
    """Run the nearest-neighbour patch visualisation and save a figure.

    The output contains four panels:
      - Test image with bounding boxes + patch-level score heatmap
      - PCA projection of the feature space (memory bank + test patches)
      - Query pixel crops from the test image
      - Nearest-normal pixel crops from the training set

    Args:
        checkpoint:  Path to a ``model.ckpt`` produced by ``engine.fit``.
        image:       Test image to inspect.
        train_dir:   Folder of normal training images (e.g. ``.../train/good``).
        top_k:       Number of most anomalous patches to visualise.
        context_px:  Side length (pixels) of each patch thumbnail.
        output:      Output PNG path (default: ``<stem>_nn_viz.png``).
        show:        Display the figure interactively.
    """
    checkpoint = Path(checkpoint)
    image      = Path(image)
    train_dir  = Path(train_dir)

    # ── load model ────────────────────────────────────────────────────────────
    print("[NN Viz] Loading checkpoint …")
    model    = Patchcore.load_from_checkpoint(str(checkpoint), weights_only=False)
    model.eval()
    m        = model.model
    mb_dtype = m.memory_bank.dtype
    print(f"  memory bank: {m.memory_bank.shape}  dtype={mb_dtype}")

    img_size = (256, 256)
    try:
        for t in model.pre_processor.transform.transforms:
            if hasattr(t, "size"):
                s = t.size
                img_size = (s[0], s[1]) if hasattr(s, "__iter__") else (s, s)
                break
    except Exception:
        pass
    print(f"  image size: {img_size}")

    # ── test image → patch embeddings ─────────────────────────────────────────
    print("[NN Viz] Extracting test patches …")
    test_tensor, test_rgb = _load(image, img_size)
    test_patches, feat_h, feat_w = _extract_patches(m, test_tensor, mb_dtype)
    N = feat_h * feat_w

    with torch.no_grad():
        patch_scores, mb_indices = m.nearest_neighbors(test_patches, n_neighbors=1)
    patch_scores = patch_scores.cpu().float()   # (N,)
    mb_indices   = mb_indices.cpu()             # (N,)

    # ── top-K selection ───────────────────────────────────────────────────────
    top_k    = min(top_k, N)
    top_flat = patch_scores.argsort(descending=True)[:top_k].tolist()
    top_rows = [idx // feat_w for idx in top_flat]
    top_cols = [idx %  feat_w for idx in top_flat]
    top_scores_list = [patch_scores[idx].item() for idx in top_flat]
    top_mb   = [int(mb_indices[idx].item()) for idx in top_flat]

    # ── score arrays ──────────────────────────────────────────────────────────
    scores_np       = patch_scores.numpy()
    score_grid_norm = scores_np.reshape(feat_h, feat_w)

    try:
        pixel_threshold = model.post_processor.pixel_threshold.item()
    except Exception:
        pixel_threshold = None

    # ── source index for pixel crops ──────────────────────────────────────────
    print("[NN Viz] Tracing memory-bank entries to training images …")
    unique_mb  = list(dict.fromkeys(top_mb))
    source_idx = _build_source_index(m, train_dir, unique_mb, img_size, mb_dtype)

    src_rgbs: dict[Path, np.ndarray] = {}
    for mb_idx, (src_path, _, _, _, _) in source_idx.items():
        if src_path not in src_rgbs:
            _, src_rgbs[src_path] = _load(src_path, img_size)

    # ── figure layout ─────────────────────────────────────────────────────────
    n_cols    = max(top_k, 2)
    stride_y  = test_rgb.shape[0] / feat_h
    stride_x  = test_rgb.shape[1] / feat_w
    fig_width = max(n_cols * 1.4, 14)

    fig = plt.figure(figsize=(fig_width, 20))
    gs  = fig.add_gridspec(
        6, n_cols,
        height_ratios=[3.5, 3.2, 0.15, 2.0, 0.15, 2.0],
        hspace=0.35, wspace=0.3,
        left=0.05, right=0.97, top=0.91, bottom=0.03,
    )

    fig.suptitle(
        f"PatchCore — Nearest-Neighbour Patch Retrieval\n"
        f"Image: {image.name}   ·   image-level score = {patch_scores.max().item():.2f}"
        f"   ·   top-{top_k} Patches",
        fontsize=15, fontweight="bold", y=0.975,
    )

    # ── row 0: test image + heatmap (centred, equal-width, tight gap) ─────────
    gs0     = gs[0, :].subgridspec(1, 2, wspace=0.08)
    ax_img  = fig.add_subplot(gs0[0, 0])
    ax_heat = fig.add_subplot(gs0[0, 1])

    ax_img.imshow(test_rgb)
    ax_img.set_title("Test Image", fontsize=13, fontweight="bold", pad=10)
    ax_img.axis("off")

    for rank, (pr, pc) in enumerate(zip(top_rows, top_cols)):
        cy   = (pr + 0.5) * stride_y
        cx   = (pc + 0.5) * stride_x
        half = context_px // 2
        rect = mpatches.FancyBboxPatch(
            (cx - half, cy - half), context_px, context_px,
            linewidth=2, edgecolor=_COLORS[rank % len(_COLORS)],
            facecolor="none", boxstyle="square,pad=0",
        )
        ax_img.add_patch(rect)
        ax_img.text(cx - half + 2, cy - half + 11, str(rank + 1),
                    color=_COLORS[rank % len(_COLORS)], fontsize=9,
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.6, pad=1, linewidth=0))

    im = ax_heat.imshow(score_grid_norm, cmap="hot", vmin=0, vmax=70)
    ax_heat.set_title("Patch Anomaly Heatmap", fontsize=13, fontweight="bold", pad=10)
    ax_heat.axis("off")
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # ── row 1: score distribution ────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[1, :])

    counts, edges = np.histogram(scores_np, bins=40)
    ax_dist.bar(
        edges[:-1], counts, width=np.diff(edges),
        align="edge", color="#4575b4", alpha=0.7, edgecolor="white",
        linewidth=0.4, label=f"all test patches (n={N})",
    )

    for rank, score in enumerate(top_scores_list):
        color = _COLORS[rank % len(_COLORS)]
        ax_dist.axvline(score, color=color, linewidth=2, alpha=0.9, zorder=3)
        ax_dist.text(
            score, 1.0, f" #{rank + 1}\nd={score:.1f}",
            color=color, fontsize=7.5, fontweight="bold",
            va="top", ha="left",
            transform=ax_dist.get_xaxis_transform(),
        )

    if pixel_threshold is not None:
        ax_dist.axvline(pixel_threshold, color="crimson", linewidth=2,
                        linestyle="--", zorder=4,
                        label=f"threshold  ({pixel_threshold:.1f})")

    ax_dist.set_xlim(0, 70)
    ax_dist.set_xlabel(
        "Distance to nearest memory-bank entry  (= anomaly score per patch)",
        fontsize=9)
    ax_dist.set_ylabel("Patch count", fontsize=9)
    ax_dist.set_title("Score Distribution", fontsize=14, fontweight="bold", pad=6)
    ax_dist.tick_params(labelsize=8)
    ax_dist.legend(fontsize=8, framealpha=0.9)

    # ── rows 2–5: section headers + crop thumbnails ───────────────────────────
    for gs_row, label in [
        (2, "Anomaly Patches – Test Image"),
        (4, "Nearest Normal Patch – Training Image"),
    ]:
        ax_hdr = fig.add_subplot(gs[gs_row, :])
        ax_hdr.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=14, fontweight="bold", transform=ax_hdr.transAxes)
        ax_hdr.axis("off")

    for rank, (pr, pc, score) in enumerate(zip(top_rows, top_cols, top_scores_list)):
        ax = fig.add_subplot(gs[3, rank])
        ax.imshow(_crop(test_rgb, pr, pc, feat_h, feat_w, context_px))
        ax.set_title(f"#{rank + 1}  score={score:.1f}", fontsize=8, pad=4)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(_COLORS[rank % len(_COLORS)])
            sp.set_linewidth(2.5)

    for rank, mb_idx in enumerate(top_mb):
        ax = fig.add_subplot(gs[5, rank])
        if mb_idx in source_idx:
            src_path, pr, pc, fh, fw = source_idx[mb_idx]
            ax.imshow(_crop(src_rgbs[src_path], pr, pc, fh, fw, context_px))
            ax.set_title(src_path.stem, fontsize=7, pad=4)
        else:
            ax.set_facecolor("#dddddd")
            ax.text(0.5, 0.5, "not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.axis("off")

    if output is None:
        output = Path(f"{image.stem}_nn_viz.png")
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[NN Viz] Saved → {output}")
    if show:
        plt.show()
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Nearest-neighbour patch visualisation for a trained PatchCore model"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to model.ckpt produced by engine.fit")
    p.add_argument("--image",      required=True,
                   help="Test image to analyse")
    p.add_argument("--train-dir",  required=True,
                   help="Folder of normal training images (e.g. .../train/good)")
    p.add_argument("--top-k",       type=int, default=3,
                   help="Number of most-anomalous patches to show (default: 5)")
    p.add_argument("--context-px",  type=int, default=48,
                   help="Crop size in pixels for each patch thumbnail (default: 48)")
    p.add_argument("--output",      type=str, default=None,
                   help="Output PNG path (default: <image_stem>_nn_viz.png)")
    p.add_argument("--show",        action="store_true",
                   help="Display the figure interactively")
    args = p.parse_args()

    visualize(
        checkpoint=args.checkpoint,
        image=args.image,
        train_dir=args.train_dir,
        top_k=args.top_k,
        context_px=args.context_px,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
