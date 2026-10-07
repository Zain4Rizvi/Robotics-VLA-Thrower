# AGENTS.md

Read this before editing the env, the expert, the policies, or the scripts. Experiment history is in `findings/`. Do not recreate status files, and do not relaunch the runs recorded there. Do not collect `data/datasets/train500` unless asked.

Repo root: `Z:\1 Github Projects\Robotics\Open Arm Folding`.

## Do not violate

- Do not change `main.py`, actuator gains, or scene physics in `v2/`. Recolor bins and balls in Python at reset.
- Right arm only. Left arm stays at `home` ctrl every step.
- GPU is an RTX 3080, 10 GB. Ampere supports bf16. Video backend is `pyav`, not torchcodec.
- Train and load scripts still cast SmolVLA to fp32 (`--policy.use_amp=false` and `.float()` in `scripts/train_smolvla.py`, `trs_so_arm100/color/train.py`, and `SmolVLAAdapter`). Hub weights load as bf16 and are cast down. A run stays fp32 until that cast is removed. Dataset arrays stay float32.
- Batch ceilings in `findings/` and `peg_socket/REPORT.md` (batch 2 once the desktop is on the card, batch 4/8 out of memory, vision batch 8) are the old 6 GB GTX 1660. On this card, pick the batch from free VRAM. `configs/train_smolvla.yaml` defaults to 8. `trs_so_arm100/color/train.py` still drops to batch 2 when used GPU memory is ≥ 800 MB; pass `--batch-size` from free VRAM.
- Nothing may be written to `C:` (~3 GB free). `setx` vars are invisible to Cursor shells, and `TEMP` defaults to `C:`. First line of every new shell:

```powershell
$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"
```

- At most two demo collectors, and never during a GPU job. Windows commit limit is ~38 GB; a third AV1 encode OOMs.
- Long jobs: `Start-Process .venv\Scripts\python.exe ... -RedirectStandardOutput/-Error ... -WindowStyle Hidden`. Killing the terminal does not stop them. Abort with `Stop-Process`, including dataloader workers.
- Do not claim a policy works without an `eval_summary.json`.

## Task

Instruction like "throw the red ball into the blue bucket" → the right OpenArm (MuJoCo) grasps that ball and throws it into that floor bin. A policy sees cameras, its own joints, and the text. It never sees ball or bin positions. The scripted expert does.

Pipeline: Gymnasium env → scripted expert → lerobot dataset → SmolVLA fine-tune → closed-loop eval.

`main.py` is the passive viewer. Scene: `v2/pedestal/throw_multi_scene.xml`. Code: `openarm_vla/`. Scripts: `scripts/`. Config: `configs/`.

## Locked interface

- 50 Hz, 20 physics substeps, dt = 0.001. Episodes cap at 400 steps (~222 in demos).
- Action (8): absolute targets for right joints 1–7 + `right_finger1_ctrl`, clipped to ctrlrange.
- State (15): right qpos (7) + qvel (7) + finger qpos (1).
- Obs: `image_front` (`headcam`; the key name is historical), `image_wrist` (`camera_wrist_right`), 256×256 RGB; `state`; `instruction`.
- Success: target ball COM inside the target bin for 0.5 s. Failures: `never_grasped`, `dropped_during_grasp`, `missed`, `wrong_bucket`, `timeout`.
- Inference: physics paused during `predict_chunk`. Chunk 16, replan every 8. Leave that loop alone.
- Gripper: finger joint `0` = closed, `-0.7854` = fully open. Grasp is friction only; `grasp_right_*` welds stay disabled. Disabling every equality also kills the finger mimic joint, so the outer finger never opens.
- Grasp point is the EE origin + `(0, 0, -0.145)` in the EE frame. The approach has to be fully open; a partial opening is narrower than the ball.
- Colors of balls, bins, target ball, and target bin are sampled independently. A landed ball is hidden from the front camera; `ball_in_bin` is authoritative.
- YAML floats must be `1.0e-4`. PyYAML reads `1e-4` as a string.
- Dataset keys, exact: `observation.images.image_front`, `observation.images.image_wrist`, `observation.state` (15, float32), `action` (8, float32), plus the task string.
- `eval_policy.py --seeds-file` must reset with the stored instruction: `env.reset(seed=s, options={"instruction": row["instruction"]})`. Template list changes do not change the scene, but they do change which phrase an old seed draws.

## SmolVLA

`SmolVLAAdapter` feeds those keys (images float 0..1) through lerobot's preprocessor, `predict_action_chunk`, then the postprocessor, and returns the first 16 of 50 steps as `(16, 8)`.

- Base `lerobot/smolvla_base` is 6-D SO-100. The adapter forces 15/8 features and empty norm stats, so base actions are meaningless here until fine-tuned. The viewer and the smoke test still load that base.
- A fine-tuned checkpoint loads its own processors, including `rename_observations_processor` (image_front → camera1, image_wrist → camera2). Adapter keys stay the same.
- `config.json` still says `observation.state` shape `[6]`. Stats are 15-D and normalization is correct. Do not "fix" the 6.
- `train_smolvla.py` runs lerobot 0.6.1 in-process. `freeze_vision_encoder` and `train_expert_only` are true (~100M of 450M). `--train-vision` trains the SigLIP tower only; `train_expert_only` still freezes the language model, and `freeze_vision_encoder=false` alone does nothing. Output dir must not exist.

`collect_demos.py`: obs before action, success episodes only, task string from `env.task["instruction"]`, writes `openarm_seeds.json`. `--out` must not already exist. Parallel runs need their own `--out` and `--seed-offset`.

## Demos on disk

- `data/datasets/train_head` — 50 expert episodes, shoulder camera. This is the set to train from.
- `data/datasets/val_head` — 20 expert episodes, same camera, no shared seeds.

The room-camera sets and the failed fine-tune checkpoints were removed. What those runs did is in `findings/`.

## Commands

`uv sync --extra train` on a fresh machine. Set the env vars above first.

```powershell
uv run python -m pytest -v
uv run python main.py
uv run python scripts/eval_expert.py
uv run python scripts/collect_demos.py --n-success 10 --max-attempts 30 --out data/datasets/NAME
uv run python scripts/dataset_stats.py --dataset data/datasets/NAME
uv run python scripts/viz_episode.py --dataset data/datasets/NAME --episode 0 --replay
uv run python scripts/train_smolvla.py --dataset-root data/datasets/train_head --val-dataset-root data/datasets/val_head --output-dir checkpoints/RUN --steps 2000 --warmup-steps 100
uv run python scripts/eval_policy.py --policy smolvla --checkpoint checkpoints/RUN/best/pretrained_model --seeds-file data/datasets/val_head/openarm_seeds.json --video-dir artifacts/RUN
uv run python scripts/vla_viewer.py
```

`vla_viewer.py` loads `lerobot/smolvla_base` unless `--checkpoint` is set. Type a sentence and Enter to replace the instruction, `r` to reset, `q` to quit.

rg/Glob skip `.venv`. Read installed lerobot 0.6.1 from disk (`lerobot.common.*` is gone).
