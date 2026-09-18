# 06 — Training

`src/engine.py`, `scripts/train.py`

## One loop, two stages

The stages differ in two things: how a batch becomes a `Prediction`, and how that
`Prediction` becomes a loss. That is what a *task* is — `StageATask` and
`StageBTask` are each about twenty lines. Everything after them — optimiser,
schedule, precision, checkpoint selection, metrics, logging — is shared, which is
why there is one `Trainer` and not two.

```python
class StageATask:
    def __call__(self, batch) -> Prediction:
        output = self.model(batch["image"], batch["prompt_ids"])
        return Prediction(
            logits=output.logits,                                    # for metrics
            target=masks_from(batch["labels"], batch["prompt_ids"] + 1),
            groups=names_of(batch["prompt_ids"]),                    # for the breakdown
            scales=output.scales,                                    # for the loss
        )

    def loss(self, prediction):
        return deep_supervision_loss(prediction.scales, prediction.target, weights)
```

`Prediction.groups[i][c]` names channel `c` of sample `i`, and it is what turns
one accumulator into "per-class Dice" for Stage A and "Dice per target class" for
Stage B without either stage owning a metrics implementation.

## Masks are built on the accelerator

`masks_from(labels, ids)` is `labels[:, None] == ids[..., None, None, None]`.
The dataset returns one `int16` label volume and a handful of integers; the
training step expands it. At 128³ with a hundred structures, materialising masks
in the dataloader would move 800 MB per sample across the process boundary.
Nothing is lost: a mask is a view of the labels, and this way it cannot drift
from them.

## The objective

```
L = Dice(logits, target) + BCEWithLogits(logits, target)
```

```yaml
loss: {name: dice_bce, lambda_dice: 1.0, lambda_bce: 1.0}
```

Stage A sums that over its three decoder scales with the weights `0.1 / 0.3 /
0.6` (`model.deep_supervision`, see [04](04_stage_a.md)); Stage B applies it at
full resolution only. The task owns its loss, which is the second and last thing
that differs between the two.

Dice supplies the gradient that matters when foreground is under 1% of the
volume; BCE keeps the background calibrated and stops Dice's plateau at
initialisation. No target-classification term, no auxiliary losses.

The loss is always computed in **float32**, even under autocast. The Dice
denominator sums one probability per voxel — 262,144 of them at 64³, two million
at 128³ — and float16 has neither the range nor the resolution to hold that sum,
so the objective would otherwise depend on the autocast dtype and the batch size.

## Schedule and precision

Each stage names its own optimiser and schedule, and both names are checked
rather than decorative — an unsupported one is an error, not a silent fallback:

```yaml
stage_a:
  optimizer: {name: adamw, lr: 0.001, weight_decay: 0.00001}
  scheduler: {name: cosine, warmup_epochs: 2}
stage_b:
  optimizer: {name: adamw, lr: 0.0003, weight_decay: 0.00001}
  scheduler: {name: cosine, warmup_epochs: 5}
```

- `scheduler.name`: `cosine` (warmup, then decay to zero) or `constant` (warmup,
  then hold).
- Gradient accumulation (`train.accum`); a partial window at the end of an epoch
  is flushed rather than dropped.
- `train.precision`: `fp32`, `bf16` or `fp16`. `bf16` is the default — it has
  float32's exponent range and needs no gradient scaler, while `fp16` leaves a
  measurable fraction of gradient elements at exactly zero on a loss dominated by
  background voxels (the trainer enables a `GradScaler` for it automatically).
  **Autocast is disabled on CPU** whatever the config says: `conv3d` has no fused
  low-precision CPU path and falls back to casting around every kernel, which
  measured about 30× slower than plain float32.
- The checkpoint is written whenever validation Dice improves.

## Early stopping

```yaml
early_stopping: {patience: 0, min_delta: 0.005}
```

Stops when validation Dice has not risen by **more than `min_delta`** for
`patience` consecutive epochs; `patience: 0` runs the full schedule. The first
value is always treated as an improvement, so a run cannot stop before it has a
baseline.

Checkpointing and stopping use different notions of "better", deliberately: any
rise saves a better checkpoint, but only a rise bigger than `min_delta` resets
the patience counter. A run that creeps up by 0.001 an epoch keeps the best
weights and still stops.

## Logging

Every run writes `metrics.jsonl` (one line per epoch), `history.json`, `best.pt`,
`last.pt` and a readable `.json` sidecar next to each checkpoint holding the
architecture, the config, the epoch and the git revision — whatever the logging
block says. The block only adds a mirror:

```yaml
logging:
  backend: wandb            # wandb | none
  wandb:
    project: relational-3d-vlm
    tags: [relational-3d-vlm, augmentation, occupancy-volume, dataset-custom]
```

Tags are pinned from this file, so a sweep cannot inherit a stale tag from an
earlier arm. `backend: none` is a no-op, and an unknown backend is an error. If
`wandb` is not installed the run prints one line and carries on with
`metrics.jsonl`.

A checkpoint carries its own architecture, so `load_model` rebuilds the right
skeleton without being told the width or resolution it was trained at.

## The four phases

Run them in order. Each one answers a question the next one depends on.

**0 — Data.** `pytest`. The direction rule, anchor ordering, prompt round-trip,
the augmentation rewrite, and the check that Stage B receives no channel
containing the target. Cheap, and it catches the failures that are invisible
later.

**1 — Stage A.** Augmentation is off for this stage by default; see
[02](02_data.md).

```bash
scripts/train.py a
```

Per-class Dice on every structure. This gates phase 4: predicted anchors are only
interesting once the segmenter is good, and the per-class table is where you see
which structures it confuses.

**2 — Overfit Stage B on one scene.**

```bash
scripts/train.py b --overfit 1 --set train.stage_b.epochs=200
```

The bug catcher, and the only phase whose *failure* is informative. Training Dice
must reach ~1.0 on a handful of examples that share one geometry. If it does not,
something structural is wrong — channel order, world coordinate orientation,
prompt indices, the decoder — and every number from phase 3 would be
uninterpretable. It proves nothing about generalisation; that is not its job.

**3 — Stage B with ground-truth anchors.**

```bash
scripts/train.py b
```

The primary measurement. Validation targets are structures never supervised as
targets, so the selection metric is itself a transfer metric.

**4 — Stage B with predicted anchors.**

```bash
scripts/evaluate.py runs/stage_b/best.pt --segmenter runs/stage_a/best.pt
```

Same weights, same prompts, anchors from Stage A instead of the label volume. The
anchors' own Dice is reported next to the result, so a drop is attributable
rather than mysterious. Passing `--segmenter` to `train.py` instead trains
against predicted anchors, which is worth doing only once the oracle number is
established.

## Reproducibility

The defaults in `configs/config.yaml` are the ones the `exp/realistic-appearance`
runs used, and `tests/test_reference_parity.py` checks that the networks still
compute what that branch computed, so a new run is comparable with an old one.

`train.seed` seeds numpy and torch; dataloader shuffling uses its own seeded
generator; synthetic scenes derive every draw from the scene seed; augmentation
draws from `(epoch, index)`. The epoch counter lives in shared memory so
`set_epoch` reaches persistent dataloader workers, which hold their own copy of
the dataset and would otherwise repeat epoch 0's poses forever.
