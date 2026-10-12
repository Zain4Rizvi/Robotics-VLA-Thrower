# Connector

The question is whether the frozen connector is why camera_text step 8000 still misses most grasps. That measurement, on seeds 30000–30009 with dimensions 6:32 held at zero, was red on green 3/10 stacks and 4/10 grasps, green on red 1/10 stacks and 3/10 grasps. Mean closest stayed 7.3 cm and 9.1 cm. Red on green never grasped 6/10. Green on red never grasped 7/10. The grasps that happened usually held. The frozen SigLIP readout places each cube to about 1.6 cm, and training that tower made the readout worse. Do not add steps to the camera_text cosine. Do not relaunch the recorded runs.

## Init

Init is a copy of `trs_so_arm100/camera_text/checkpoints/checkpoints/008000/pretrained_model`, stored at `trs_so_arm100/connector/init`. Train from that copy. Fresh Adam. Do not resume its optimizer, and do not load its `training_state`.

Those weights already grasp on some seeds from the cameras and the two sentences. The connector stayed frozen for all 8000 of those steps, so it never had to carry cube position into the action expert. This run asks whether training the connector can turn the frozen image tokens into a grasp. Step 0 is a new rollout of the copy before any optimizer step. The camera_text table was 3/10 and 1/10 stacks, 4/10 and 3/10 grasps. Noise makes a new draw differ. Record step 0 before training.

## What trains

`freeze_vision_encoder` stays true. `train_expert_only` stays true. Do not set `train_expert_only` false. That would unfreeze the text model. After the policy is built, and before the optimizer is built, set every connector parameter `requires_grad` true. `policy.train()` puts the VLM in eval and does not by itself clear `requires_grad`. Wrap `train` so a later call still leaves the connector trainable, the vision tower frozen, and the text layers frozen.

Do not call `sentence/cont/harness.py` `assert_schedule` or `execute`. `assert_schedule` aborts if any connector parameter is trainable, and it aborts if any VLM parameter is trainable. Do not change `color/train.py` `assert_expert_only`. It still refuses a decay other than 2000. This script checks its own schedule.

Before the first optimizer step, print the trainable count and abort unless all of these are true: at least one connector parameter is trainable, no vision-tower parameter is trainable, no text-layer parameter is trainable, the action expert is trainable, and `state_proj` is trainable. On the first `optimizer.zero_grad`, abort if every connector gradient is `None` or zero.

## State

Dimensions 0:6 are the six joint angles. Dimensions 6:32 stay zero at train time and at rollout. That includes xy and the probe logits in dimensions 10:12. The swap dataset's `xy.npz` is not written into `observation.state`. Closest at rollout is the live sit pose. The train step and every rollout chunk abort if any value in dimensions 6:32 is nonzero at `state_proj`. The zero-tail check is imported from `camera_text/train.py`.

Read-only data is `trs_so_arm100/color/swap/datasets/train` and `val`: 40 train and 16 val episodes of each sentence, `stack_seeds.json`. Do not collect a new dataset. Do not move or rewrite those datasets or the camera_text checkpoints.

Sentences:

- stack the red cube on the green cube
- stack the green cube on the red cube

fp32, `use_amp` false, dataset arrays float32, video backend `pyav`.

## Budget

2000 steps, fresh Adam, new cosine, peak learning rate `1.0e-4`, warmup 100, decay over those 2000 steps, floor `2.5e-6`. Save a `pretrained_model` every 500 steps, including a copy of the init weights at step 0, and keep every `model.safetensors`. Pruning `training_state` folders is fine. Train loss every step. Val loss every 500 steps.

After the first 500 steps, convert that wall clock to minutes per 1000 steps. Slower than 45 minutes per 1000 steps cuts the run to 1000 steps, retargets the cosine, and drops the 1500 and 2000 rollouts. Write the cut in the report.

Batch starts at 8. If the first steps leave at least 1 GB free, stop, delete the train output directory, and try batch 16 once. On CUDA OOM, delete that output directory, halve the batch, and retry once. A second OOM stops the run.

One GPU job. If `nvidia-smi` already shows a python compute process, do not start another. `Test-Path Z:\` before the first shell. If it is true, use the AGENTS.md prelude. If it is false, use the offline prelude in `sentence/cont/PLAN.md`. Do not write caches or temp files to `C:`.

## Rollout

Seeds 30000–30009, both sentences, disjoint from train and val. Steps 0, 500, 1000, 1500, and 2000. Drop 1500 and 2000 if the run was cut to 1000. Step 0 runs before training.

Same control as camera_text: 30 Hz, replan every 10, physics paused while the chunk is computed, pitch sign as in `view.py`. Success is the source cube on the target cube, in contact, for 0.5 s, with the target cube still on the floor. Recolor stays geom rgba only. Do not change masses, friction, gains, or the floor. The floor geom stays `groundplane`.

For each sentence: stacks/10, grasps/10, drops, closest best, closest mean, and a failure split (never grasped, grasped then dropped, reached the pose but did not hold 0.5 s). Closest is the source-cube center to the sit pose, in centimeters. One video per sentence per checkpoint, top camera with the wrist inset, seed 30000. Pad the step number. Keep a video when it is a miss. Stills of the last step only: a success on that sentence if any of the ten seeds stacked, otherwise the closest miss.

## Do not touch

Do not modify, delete, rename, or write checkpoints into `best/`, `sentence/cont/run1/`, `sentence/use_note/`, `color/swap/`, `camera_text/` other than reading its step-8000 weights and its `REPORT.md`, `red_green_view.py`, `view.py`, or `playground/`. Do not point the web app at this checkpoint.
