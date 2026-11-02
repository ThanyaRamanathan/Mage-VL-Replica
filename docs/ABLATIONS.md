# Validation-only selector ablations

These ablations must write to separate directories and use `SKIP_TEST=1`.
Choose one setting from validation results, then evaluate the held-out test set
once with that frozen choice.

## Codec time scale and motion weight

Build distinct caches so cache metadata records every signal choice:

```bash
for aggregation in instantaneous max_interval mean_interval; do
    for weight in 0.25 0.50 0.75; do
        output="/scratch/path/codec_${aggregation}_mw${weight}"
        qsub -v "CONDA_ENV=surgical,MANIFEST=/scratch/path/cataract101.csv,VIDEOS=/scratch/path/cataract101/cataract-101,OUTPUT=${output},MOTION_WEIGHT=${weight},TEMPORAL_AGGREGATION=${aggregation}" cache_cataract101_codec.pbs
    done
done
```

Train codec validation runs at 1,568 and 3,136 tokens. Use one seed for the
screen, then confirm the chosen setting across all three training seeds.

## Anchor fraction

Screen `ANCHOR_FRACTION=0.25,0.5,0.75` with the selected codec cache. The anchor
audit must be rerun for each fraction because the allowed dense-anchor count
changes.

## Score normalization

Screen both supported modes:

```text
SCORE_NORMALIZATION=none
SCORE_NORMALIZATION=per_frame_rank
```

`per_frame_rank` removes arbitrary offsets and scales independently for every
frame. It is a symmetry control, not a replacement for clip-calibrated clinical
training. The v0.3 clinical scorer should use `CLINICAL_TRAINING_MODE=clip` and
`CLINICAL_BATCH_SIZE=2` so cross-frame offsets affect the supervised loss.

## Example validation-only classifier submission

```bash
qsub -v SELECTOR=codec,BUDGET=1568,SEED=17,CONFIG=configs/v03/codec.yaml,MANIFEST=/scratch/path/cataract101.csv,CODEC_SCORES=/scratch/path/codec_max_interval_mw0.50,ANCHOR_FRACTION=0.5,SCORE_NORMALIZATION=per_frame_rank,SKIP_TEST=1,OUTPUT_ROOT=/scratch/path/ablations train_surgical.pbs
```

The output contains `validation_selection.json` and no test metrics.
