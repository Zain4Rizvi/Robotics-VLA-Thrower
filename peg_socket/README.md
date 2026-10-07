# Peg in the hole

The record of what was run, and what the arm did, is [REPORT.md](REPORT.md). Throw-task history stays in `findings/`.

The user says "place the peg in the hole" and the right arm does it. The policy sees the text, its own joints, the face camera, and the right wrist camera. It does not see the peg position or the socket position. The scripted expert does. The expert is for demonstrations. It is not the result.

## Contract

- Scene: [peg_socket_scene.xml](peg_socket_scene.xml), a copy of `v2/pedestal/peg_socket_scene.xml`. Do not edit `v2/`. The copy is the file with the socket free joint.
- Start pose: MuJoCo key `home`. Both arms at the shared home pose, peg upright on the table.
- Right arm only. Left arm holds the home ctrl every step.
- Action (8): absolute targets for right joints 1–7 and `right_finger1_ctrl`.
- State (15): right qpos (7) + qvel (7) + finger qpos (1).
- Cameras: `tablecam` is the face view, stored as `observation.images.image_front`. `camera_wrist_right` is `observation.images.image_wrist`. 256×256 RGB. `frontcam` is unused.
- Instruction, every episode: `place the peg in the hole`.
- Grasp is friction only. `grasp_right_peg` stays disabled. The approach has to be fully open.
- Success: peg center stays inside the socket opening for 0.5 s. Failures: `never_grasped`, `dropped`, `missed`, `timeout`.
- GPU is an RTX 3080, 10 GB. It supports bf16. Train and load scripts still run fp32 until the `.float()` cast is removed. Nothing written to `C:`. First line of every shell:

```powershell
$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"
```

- At most two demo collectors, and never while a GPU job is running.

## Code

- [peg_expert.py](peg_expert.py) plans the insertion from the true peg and socket poses.
- [eval_peg_expert.py](eval_peg_expert.py) draws legal layouts and scores insertions. The 18/20 run is in the report.
- [collect_peg_demos.py](collect_peg_demos.py) writes success episodes. `--out` must not already exist.

Datasets and the face-camera checkpoint are named in the report. They are not in this folder.
