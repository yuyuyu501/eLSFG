# eLSFG

eLSFG is a Windows desktop prototype for low-latency game super resolution.

The current MVP focuses on 480p to 1440p super resolution with a lightweight
Transformer model. Performance benchmarks should be treated as valid only when
they are run on the RTX 2080 Ti training machine.

## Setup

Create the conda environment inside the project, not in base:

```powershell
conda env create -f environment.yml
conda activate elsfg
python scripts/check_env.py
```

If the environment already exists:

```powershell
conda activate elsfg
python -m pip install -r requirements.txt
python scripts/check_env.py
```

## Run The App

```powershell
.\run_app.ps1
```

Select the `AI Quality` profile to use the packaged Transformer checkpoint at
`checkpoints/elsfg_sr_detail_aware.pt`.

## Test

```powershell
python -m unittest discover -s tests -v
```

On the RTX 2080 Ti machine, run the tensor benchmark:

```bash
python scripts/benchmark_sr_tensor.py \
  --variant detail_aware \
  --model-dim 12 \
  --model-depth 1 \
  --model-heads 3 \
  --width 854 \
  --height 480 \
  --target-width 2560 \
  --target-height 1440 \
  --runs 100 \
  --warmup 20 \
  --channels-last
```

The target is at least 120 FPS with less than 4096 MB peak VRAM.
