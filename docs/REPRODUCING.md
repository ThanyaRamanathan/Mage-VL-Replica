# Reproduction guide

## Requirements

- A Linux HPC system with PBS, CUDA, FFmpeg/PyAV, and an NVIDIA GPU.
- Python 3.10+ with the dependencies in `pyproject.toml`.
- Accepted access to Cataract-101 and compliance with its terms.
- Approximately 15 GB for source videos, extracted samples, score caches,
  checkpoints, and logs.

The checked-in PBS files use NSCC-style module names and queue syntax. Adapt
the `#PBS` resources and `module load` lines for another cluster. All data paths
can be overridden with `qsub -v`; defaults derive from
`/scratch/users/ntu/$USER` through `SCRATCH_ROOT`.

## 1. Environment and smoke test

From the repository root:

```bash
qsub -v ENV_NAME=surgical setup_surgical_env.pbs
qsub -v CONDA_ENV=surgical smoke_surgical.pbs
```

The original experiment used an NVIDIA A100-SXM4-40GB.

## 2. Acquire and index Cataract-101

Review the dataset terms before enabling the downloader:

```bash
qsub -v CONDA_ENV=surgical,SCRATCH_ROOT=/your/scratch/path prepare_cataract101.pbs
```

The converter samples the original H.264 streams at 2 fps and writes a frozen
procedure-level 71/15/15 split with split seed 17. `cataract101.splits.json` in
this repository records the experiment's procedure IDs and source hashes.

## 3. Build native codec scores

```bash
qsub -v CONDA_ENV=surgical,SCRATCH_ROOT=/your/scratch/path cache_cataract101_codec.pbs
```

This produces per-video motion-vector and decoded residual-proxy score maps.
It does not expose or claim bit-exact H.264 transform coefficients.

## 4. Train each clinical importance scorer and cache its maps

```bash
for seed in 17 29 43; do
    qsub -v STAGE=clinical-cache,SEED="$seed",CONDA_ENV=surgical,SCRATCH_ROOT=/your/scratch/path run_cataract101_gpu.pbs
done
```

## 5. Train the 27 matched-budget classifiers

`train_surgical.pbs` runs one configuration per job. Independent jobs avoid
putting nine runs behind one 48-hour wall-time limit:

```bash
for seed in 17 29 43; do
    for selector in uniform codec clinical; do
        for budget in 1568 3136 6272; do
            extra=""
            if [ "$selector" = codec ]; then
                extra=",CODEC_SCORES=/your/scratch/path/cataract101_codec_scores"
            elif [ "$selector" = clinical ]; then
                extra=",CODEC_SCORES=/your/scratch/path/cataract101_codec_scores,CLINICAL_SCORES=/your/scratch/path/cataract101_clinical_scores_seed${seed}"
            fi
            qsub -v "SELECTOR=${selector},BUDGET=${budget},SEED=${seed},CONDA_ENV=surgical,MANIFEST=/your/scratch/path/cataract101.csv,OUTPUT_ROOT=$PWD/cataract101_runs${extra}" train_surgical.pbs
        done
    done
done
```

Each canonical run directory contains `results.json` and a locally ignored
`best.pt`. A complete grid contains 27 result JSONs.

## 6. Rebuild tables and figures

On any machine with Python, NumPy, and Pillow:

```bash
python scripts/analyze_results.py
python scripts/qualitative_selectors.py
```

The first command validates the exact 3×3×3 design before writing the report
CSVs and SVG figures. The second command requires local frames and selector
caches; generated medical-frame JPEGs are ignored by Git.
