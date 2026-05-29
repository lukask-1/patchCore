#!/usr/bin/env python3
"""
PatchCore on MVTec AD – powered by anomalib.
https://arxiv.org/abs/2106.08265

Training:
  python patchcore.py --category bottle
  python patchcore.py --category all
  python patchcore.py --backbone resnet18 --coreset-ratio 0.05

Inference on arbitrary images:
  python patchcore.py --infer --checkpoint results/bottle/... --image path/to/img.png
  python patchcore.py --infer --checkpoint results/bottle/... --image path/to/folder/
"""

import argparse
from pathlib import Path

import anomalib
import matplotlib.pyplot as plt
import torch
from anomalib.data import MVTecAD
from anomalib.data.predict import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Required so anomalib checkpoints can be loaded with weights_only safety
torch.serialization.add_safe_globals([anomalib.PrecisionType])

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def predict_image(
    checkpoint: str | Path,
    image: str | Path,
    output: str | Path | None = None,
    show: bool = False,
) -> list[dict]:
    """Load a trained PatchCore checkpoint and run inference on an image or folder.

    Passing a folder scores all images inside in a single model-load, which
    avoids re-paying the ~5 s startup cost per image.

    Args:
        checkpoint: Path to a ``model.ckpt`` file produced by ``engine.fit``.
        image:      Path to a single image file or a directory of images.
        output:     Where to write visualisation(s).
                    - Single image → path to output PNG file.
                    - Folder input → path to output directory (created if absent).
                    Defaults to ``<stem>_anomaly.png`` in the current directory
                    (single image) or ``./anomaly_results/`` (folder).
        show:       If True, display each result interactively via matplotlib.

    Returns:
        List of dicts, one per image:
        ``{"image_path", "pred_score", "is_anomalous", "anomaly_map"}``
    """
    input_path = Path(image)
    is_folder = input_path.is_dir()

    # Resolve output destination
    if output:
        out_base = Path(output)
    elif is_folder:
        out_base = Path("anomaly_results")
    else:
        out_base = None  # resolved per-image below

    if is_folder and out_base:
        out_base.mkdir(parents=True, exist_ok=True)

    model = Patchcore.load_from_checkpoint(str(checkpoint), weights_only=False)
    dataset = PredictDataset(path=input_path)
    engine = Engine()

    n_images = len(dataset)
    print(f"[PatchCore] Scoring {n_images} image(s) ...")
    batches = engine.predict(model=model, dataset=dataset)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    results = []
    for batch in batches:
        for i in range(len(batch.image)):
            img_path = Path(batch.image_path[i])
            score = float(batch.pred_score[i])
            is_anomalous = bool(batch.pred_label[i])
            amap = batch.anomaly_map[i].cpu().numpy().squeeze()
            mask = batch.pred_mask[i].cpu().numpy().squeeze().astype(bool)

            results.append({
                "image_path": str(img_path),
                "pred_score": score,
                "is_anomalous": is_anomalous,
                "anomaly_map": amap,
            })

            rgb = (batch.image[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            # Red overlay of predicted mask on original
            overlay = rgb.copy()
            overlay[mask, 0] = 0.9
            overlay[mask, 1] *= 0.3
            overlay[mask, 2] *= 0.3

            label_str = "ANOMALOUS" if is_anomalous else "NORMAL"
            _, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(rgb);              axes[0].set_title("Input");                                   axes[0].axis("off")
            im = axes[1].imshow(amap, cmap="jet", vmin=0, vmax=0.9)
            axes[1].set_title(f"Anomaly map  (score={score:.3f}  {label_str})"); axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            axes[2].imshow(mask, cmap="gray"); axes[2].set_title("Predicted mask");                         axes[2].axis("off")
            axes[3].imshow(overlay);           axes[3].set_title("Mask overlay");                           axes[3].axis("off")
            plt.tight_layout()

            if is_folder and out_base:
                out_path = out_base / f"{img_path.stem}_anomaly.png"
            elif out_base:
                out_path = out_base  # caller supplied explicit file path
            else:
                out_path = Path(f"{img_path.stem}_anomaly.png")

            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  {img_path.name}: score={score:.4f}  {label_str}  → {out_path}")

            if show:
                plt.show()
            plt.close()

    return results


def train(args: argparse.Namespace) -> None:
    categories = CATEGORIES if args.category == "all" else [args.category]

    for category in categories:
        print(f"\n{'='*55}\n  Category: {category}\n{'='*55}")

        datamodule = MVTecAD(
            root=args.data_root,
            category=category,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = Patchcore(
            backbone=args.backbone,
            coreset_sampling_ratio=args.coreset_ratio,
        )

        engine = Engine(default_root_dir=Path(args.output_dir) / category)
        engine.fit(model=model, datamodule=datamodule)
        engine.test(model=model, datamodule=datamodule)


def main() -> None:
    p = argparse.ArgumentParser(description="PatchCore anomaly detection on MVTec AD")

    # ── inference mode ────────────────────────────────────────────────────────
    p.add_argument("--infer", action="store_true",
                   help="Run inference on an arbitrary image instead of training")
    p.add_argument("--checkpoint", type=str,
                   help="Path to model.ckpt (required with --infer)")
    p.add_argument("--image", type=str,
                   help="Image file or folder to score (required with --infer). "
                        "Passing a folder scores all images in one model-load, "
                        "which is much faster than repeated single-image calls.")
    p.add_argument("--output", type=str, default=None,
                   help="Where to save the visualisation PNG (--infer only)")
    p.add_argument("--show", action="store_true",
                   help="Display result interactively (--infer only)")

    # ── training mode ─────────────────────────────────────────────────────────
    p.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    p.add_argument("--category", default="bottle", choices=CATEGORIES + ["all"])
    p.add_argument("--backbone", default="wide_resnet50_2",
                   choices=["wide_resnet50_2", "resnet50", "resnet18"])
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", default="results")

    args = p.parse_args()

    if args.infer:
        if not args.checkpoint or not args.image:
            p.error("--infer requires both --checkpoint and --image")
        predict_image(args.checkpoint, args.image, output=args.output, show=args.show)
    else:
        train(args)


if __name__ == "__main__":
    main()
