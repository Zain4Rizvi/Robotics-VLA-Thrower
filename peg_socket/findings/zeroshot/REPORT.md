# zero-shot

Status: done

## Result

`lerobot/smolvla_base` finished 20 legal layouts without a crash. `eval_summary.json` records 0 insertions and 0 grasps. Every episode failed as `never_grasped`. The closest the grasp point came to the peg center was 0.04782093884098865 m. That is the baseline. The policy did not place the peg.

## Process

From the repo root, with `HF_HOME`, `UV_CACHE_DIR`, `TEMP`, `TMP`, `TORCH_HOME`, and `XDG_CACHE_HOME` on `Z:`, and `MUJOCO_GL=glfw`:

```
.venv\Scripts\python.exe -u peg_socket\zeroshot_eval.py
```

Checkpoint `lerobot/smolvla_base` through `SmolVLAAdapter`, fp32 on CUDA. Instruction `place the peg in the hole`. Seed 0. Chunk 16, replan every 8, physics paused during `predict_chunk`. Episode cap 400 control steps. Right-arm targets clipped to ctrlrange. Left arm held at home ctrl. `grasp_right_peg` stayed off. `tablecam` is `image_front` and `camera_wrist_right` is `image_wrist`. `frontcam` was not used.

Draws are uniform in x `[0.20, 0.50]` and y `[-0.46, 0.02]`. A draw is kept only when the peg (radius 0.014 m) and the socket (outer radius 0.030 m) sit on the table, clear the pedestal collision box by 0.005 m, are at least 0.12 m apart, project inside `tablecam` with a 16 px margin, and touch nothing except `table_top`. 12 draws were rejected (`too_close` 4, `gripper` 4, `contact` 4). The home keyframe passes the same test. The scene copy was not edited.

## Numbers

`peg_socket/findings/zeroshot/eval_summary.json`

```
n=20
insertions=0
grasp_count=0
min_hand_to_peg_m=0.04782093884098865
median_episode_min_hand_to_peg_m=0.1057739570414589
```

Per-episode minimum grasp-point to peg-center distance, in meters, episodes 0–19:

```
0.15317539137189184
0.11298164881711509
0.10748568727482355
0.05025881412063768
0.05174937080342202
0.10406222680809424
0.0979302709833202
0.17404934015886558
0.23068125615479299
0.08770192881076863
0.05226135024725116
0.08423837550581202
0.16868490104659817
0.06412267033098883
0.18064425534777068
0.20464434927388964
0.14473334076644656
0.1958347666990083
0.10389083719962262
0.04782093884098865
```

All 20 failures are `never_grasped`.

## Failures other agents should know

Missing every peg is the baseline, not a crash. Do not retune this run, and do not point a later run at throw checkpoints or `data/datasets/train_head`.

The home peg sits inside the axis-aligned box of the finger meshes and still does not touch them. An exclusion written from that box rejects the home pose. This run rejects a draw only when the peg or socket contacts a geom other than `table_top`. Base actions are meaningless.

## Sources

- `peg_socket/findings/zeroshot/eval_summary.json`
- `peg_socket/findings/zeroshot/zeroshot_stdout.txt`
- `peg_socket/findings/zeroshot/zeroshot_stderr.txt`
- `peg_socket/zeroshot_eval.py`
