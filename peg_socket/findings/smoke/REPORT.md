# smoke load

Status: done

## Result

`lerobot/smolvla_base` loaded in fp32 on CUDA and stepped 4 action chunks on the peg-socket scene copy after a reset to keyframe `home` (key 0). It did not crash. `tablecam` is `image_front` and `camera_wrist_right` is `image_wrist`, both `(256, 256, 3)` uint8. State is `(15,)` float32. Each chunk is `(16, 8)` float32. Step 4 may start. This run does not say the policy placed the peg. Base actions are meaningless until fine-tuned.

## Process

From the repo root, with `HF_HOME`, `UV_CACHE_DIR`, `TEMP`, `TMP`, `TORCH_HOME`, and `XDG_CACHE_HOME` on `Z:`, and `MUJOCO_GL=glfw`:

```
.venv\Scripts\python.exe -u peg_socket\smoke_load.py
```

`peg_socket/smoke_load.py` loads `peg_socket/peg_socket_scene.xml`, resets with `mj_resetDataKeyframe` on key 0, forces `grasp_right_peg` inactive, and holds the left arm at home ctrl on every control tick. It renders `tablecam` and `camera_wrist_right` at 256×256. `frontcam` is not read. The instruction is `place the peg in the hole`. `SmolVLAAdapter` loads checkpoint `lerobot/smolvla_base` (no fine-tuned checkpoint). Physics is paused during `predict_chunk`. Each of 4 chunks is clipped to the right-arm ctrlrange and applied for 16 control ticks of 20 substeps. The scene copy was not edited.

## Numbers

| Key | Source | dtype | shape |
|---|---|---|---|
| `image_front` | `tablecam` → `observation.images.image_front` | uint8 | (256, 256, 3) |
| `image_wrist` | `camera_wrist_right` → `observation.images.image_wrist` | uint8 | (256, 256, 3) |
| `state` | right qpos (7) + qvel (7) + finger qpos (1) | float32 | (15,) |
| `instruction` | fixed sentence | str | `place the peg in the hole` |
| action | `SmolVLAAdapter.predict_chunk` | float32 | (16, 8) |

Log:

```
key_name=home key_id=0 nq=32
grasp_right_peg_active=0
qpos_max_abs_err=0.000e+00
camera tablecam id=0
camera camera_wrist_right id=3
camera frontcam id=1
checkpoint=lerobot/smolvla_base
policy_dtype=torch.float32 device=cuda:0
instruction='place the peg in the hole'
obs_keys=image_front,image_wrist,state,instruction
obs.image_front shape=(256, 256, 3) dtype=uint8
obs.image_wrist shape=(256, 256, 3) dtype=uint8
obs.state shape=(15,) dtype=float32
map observation.images.image_front=tablecam
map observation.images.image_wrist=camera_wrist_right
frontcam_used=0
chunk=0 action_shape=(16, 8) dtype=float32 finite=1 min=-0.4485 max=1.5173
chunk=0 stepped=16 qpos_finite=1
chunk=1 action_shape=(16, 8) dtype=float32 finite=1 min=-0.7923 max=1.1489
chunk=1 stepped=16 qpos_finite=1
chunk=2 action_shape=(16, 8) dtype=float32 finite=1 min=-2.8063 max=0.5537
chunk=2 stepped=16 qpos_finite=1
chunk=3 action_shape=(16, 8) dtype=float32 finite=1 min=-2.2062 max=1.7737
chunk=3 stepped=16 qpos_finite=1
chunks=4 crashed=0
```

## Failures other agents should know

The first launch exited before any chunk because `SmolVLAPolicy` has no `.dtype` attribute. The rerun above is the one that stepped. Do not read `policy.dtype`; the parameter dtype is `torch.float32`. Base actions are meaningless. Do not point step 4 at throw-run checkpoints or `data/datasets/train_head`.

## Sources

- `peg_socket/smoke_load.py`
- `peg_socket/findings/smoke/smoke_stdout.txt`
- `peg_socket/findings/smoke/smoke_stderr.txt`
