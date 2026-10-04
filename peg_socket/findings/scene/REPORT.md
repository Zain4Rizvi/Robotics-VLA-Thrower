# scene and start pose

Status: done

## Result

Reset to keyframe `home` (key 0) matches the stored qpos with max abs error 0. Both arms are in that pose, and `tablecam` plus `camera_wrist_right` are saved at 256×256. Moving `socket_free` qpos by +0.08 m moves the socket body by 0.08 m in x. Steps 1 and 2 may start.

## Process

Copied `v2/pedestal/peg_socket_scene.xml` to `peg_socket/peg_socket_scene.xml` and edited only the copy. Added `<joint name="socket_free" type="free"/>` on the socket body and extended the `home` keyframe qpos with the socket pose (`0.36 -0.26 0.40 1 0 0 0`, qpos address 25). Include paths on the copy point at `../v2/pedestal/openarm_pedestal.xml` and `../v2/assets` so the file loads from `peg_socket/`. `v2/` was not edited.

From the repo root, after the cache env vars:

```
uv run python peg_socket/render_home.py
```

`peg_socket/render_home.py` loads the copy, calls `mj_resetDataKeyframe` on key 0, shifts `socket_free`, resets again, then writes the two frames.

## Numbers

| Check | Value |
|---|---|
| key | `home`, id 0 |
| nq | 32 |
| qpos vs keyframe | max abs error 0 |
| left arm qpos[0:9] vs home | max abs error 0 |
| right arm qpos[9:18] vs home | max abs error 0 |
| socket home xpos | 0.36, -0.26, 0.40 |
| socket xpos after qpos[25] += 0.08 | 0.44, -0.26, 0.40 (delta 0.08, 0, 0) |
| frames | 256×256×3 uint8 |

Log:

```
key_name=home key_id=0 nq=32 nv=30
qpos_max_abs_err=0.000e+00 left_arm_err=0.000e+00 right_arm_err=0.000e+00
socket_free_qposadr=25 home_xpos=[ 0.36 -0.26  0.4 ]
moved_xpos=[ 0.44 -0.26  0.4 ] delta=[0.08 0.   0.  ]
restored_xpos=[ 0.36 -0.26  0.4 ] restore_err=0.000e+00
tablecam shape=(256, 256, 3) dtype=uint8 mean=202.97
camera_wrist_right shape=(256, 256, 3) dtype=uint8 mean=81.46
```

## Failures other agents should know

Nothing failed.

## Sources

- `peg_socket/peg_socket_scene.xml`
- `peg_socket/render_home.py`
- `peg_socket/findings/scene/reset.log`
- `peg_socket/findings/scene/tablecam.png`
- `peg_socket/findings/scene/wrist.png`
