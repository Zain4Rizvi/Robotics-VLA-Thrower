# Chunk playback

Plan only. No run has been done. Results go in `REPORT.md` in this folder when the run finishes. Leave this file as the plan.

## Question

The face-probe policy is shaky because playback cuts its prediction short. The checkpoint was trained to emit 50 joint targets, one second at 50 Hz (`chunk_size` and `n_action_steps` in `checkpoints/peg_face_probe/best/pretrained_model/config.json`). `SmolVLAAdapter` keeps 16 of those 50, and `zeroshot_eval.rollout` executes 8 and then predicts again. Each new prediction is a new noise sample, and the joint target yanks.

This experiment asks whether playing more of each prediction makes the arm move smoothly. Accuracy is recorded and is not the pass condition. A smooth miss is a successful answer to this question. A chunk that is still shaky inside its own 50 steps is a failed answer, and it means a longer playback will not help.

Do not fine-tune. Do not collect demos. Do not edit `v2/`, the throw eval loop, or the class default `SmolVLAAdapter.chunk_size`.

## What to run

Same weights, same scenes, three playbacks. Checkpoint:

`checkpoints/peg_face_probe/best/pretrained_model`

Seeds, in order, from `data/datasets/peg_face_val/openarm_seeds.json`:

30003, 30006, 30013, 30014, 30032.

Instruction `place the peg in the hole`. Episode cap 900 control steps, so the clips are the same length as the face-probe videos. Physics paused during `predict_chunk`. Right arm only. Left arm at home. `grasp_right_peg` off. fp32. Face camera (`tablecam`) in the video.

| Folder | Keep from the prediction | Steps executed before the next prediction |
|---|---|---|
| `exec8/` | 16 | 8 |
| `exec16/` | 16 | 16 |
| `exec50/` | 50 | 50 |

`exec8` is the face-probe controller. It exists so this folder stands alone. Set the keep-length on the adapter instance for that run (`policy.chunk_size = 50` only for `exec50`). Set how many steps are executed before the next `predict_chunk` for that run. Do not change `openarm_vla/policies/smolvla.py` or `scripts/eval_policy.py`.

One face-camera mp4 per seed in each folder:

`exec8/videos/seed30003.mp4`, and the same pattern for the other seeds and the other two folders.

Also write `eval_summary.json` in each folder: insertions, grasps, closest hand-to-peg, median of the per-episode closest, and the jump numbers below.

## Plots

At every control step record the joint-target step, the L2 norm of the change from the previous target. Mark a step as a boundary when it is the first target of a new prediction.

Save `jumps.png` in this folder:

- Three groups, one per playback.
- Two bars in each group: median step size inside a prediction, and median step size at a boundary.

Save `trace_seed30006.png`: the step size over time for seed 30006 under all three playbacks, so a boundary yank is visible as a spike.

If the arm is smooth, the boundary bar falls from `exec8` to `exec50` and the trace loses its spikes. If `exec50` is still spiky inside the 50 steps, say so in the report. That is the result.

Also save `closest.png`: closest hand-to-peg, in centimeters, for the three playbacks (the minimum, and the median across the 5 seeds). This plot is context. It does not decide the question.

## Report

Write `REPORT.md` here when the videos and plots exist. Result first: did the boundary jumps shrink, and do the videos look steady. Then the three commands, then the jump table and the closest-distance table. Link every video. Quote `eval_summary.json`. A falling jump with the peg still on the table is a pass for this question.

## Do not

- Do not train, and do not start a demo collector.
- Do not point this at `data/datasets/peg_train`, `peg_val`, or any throw checkpoint.
- Do not blend an old chunk into a new one. This test only changes how much of one prediction is played.
- Do not claim the peg was inserted unless `eval_summary.json` says so.
