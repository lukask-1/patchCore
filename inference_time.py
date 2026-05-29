#!/usr/bin/env python3
"""
Measures the wall-clock time of each phase in a PatchCore inference call:
  - Package imports
  - Checkpoint loading
  - Engine initialisation
  - First inference  (includes MPS/CUDA warm-up JIT)
  - Second inference (steady-state, model already loaded)
"""

import time

CKPT  = "results/bottle/Patchcore/MVTecAD/bottle/v1/weights/lightning/model.ckpt"
IMG_1 = "data/mvtec_anomaly_detection/bottle/test/broken_large/000.png"
IMG_2 = "data/mvtec_anomaly_detection/bottle/test/broken_large/001.png"

# ── 1. Imports ────────────────────────────────────────────────────────────────
t0 = time.perf_counter()

import anomalib
import torch
from anomalib.data.predict import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

t1 = time.perf_counter()

# ── 2. Checkpoint loading ─────────────────────────────────────────────────────
torch.serialization.add_safe_globals([anomalib.PrecisionType])
model = Patchcore.load_from_checkpoint(CKPT, weights_only=False)

t2 = time.perf_counter()

# ── 3. Engine initialisation ──────────────────────────────────────────────────
engine = Engine()

t3 = time.perf_counter()

# ── 4. First inference (device warm-up included) ──────────────────────────────
engine.predict(model=model, dataset=PredictDataset(path=IMG_1))

t4 = time.perf_counter()

# ── 5. Second inference (steady-state) ────────────────────────────────────────
engine.predict(model=model, dataset=PredictDataset(path=IMG_2))

t5 = time.perf_counter()

# ── Results ───────────────────────────────────────────────────────────────────
total = t5 - t0
rows = [
    ("Imports",                    t1 - t0),
    ("Checkpoint load",            t2 - t1),
    ("Engine init",                t3 - t2),
    ("1st inference (warm-up)",    t4 - t3),
    ("2nd inference (steady)",     t5 - t4),
]

print(f"\n{'Phase':<30} {'Time':>8}  {'Share':>7}")
print("-" * 48)
for name, dt in rows:
    print(f"{name:<30} {dt:>7.2f}s  {dt/total*100:>6.1f}%")
print("-" * 48)
print(f"{'Total':<30} {total:>7.2f}s  100.0%")
