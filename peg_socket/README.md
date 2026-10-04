# Peg in the hole

End state: the user says "place the peg in the hole" and the right arm does it. The policy sees the text, its own joints, the face camera, and the right wrist camera. It does not see the peg position or the socket position.

The scripted expert does see those positions. It exists to collect demonstrations. It is not the result.

Throw-task history stays in `findings/`. This task logs only under `peg_socket/findings/`.

## Why this task

The throw runs failed because the network never reported which ball the sentence named. Here there is one peg and one socket, and the sentence does not change. What still has to work is seeing where they are.

The hole is tight. The peg radius is 1.4 cm and the socket's inner radius is 2.0 cm, so the peg center has to be within about 6 mm of the hole center. On the throw scenes, the best localization was 9 cm, and SmolVLM2 pointed at the ball in 0 of 20 frames (`findings/vision_xy/REPORT.md`). Step 5 exists so we do not fine-tune an action head on top of that failure again.

## Contract

- Scene file for this work: a copy of `v2/pedestal/peg_socket_scene.xml`, living under `peg_socket/`. Do not edit `v2/`. The socket in `v2` has no free joint, so it cannot be moved. The copy is where that joint gets added.
- Start pose: MuJoCo key 0, the `home` keyframe. Both arms at the shared home pose, peg upright on the table.
- Right arm only. Left arm holds the home ctrl every step.
- Action (8): absolute targets for right joints 1–7 and `right_finger1_ctrl`.
- State (15): right qpos (7) + qvel (7) + finger qpos (1).
- Cameras: `tablecam` is the face view, between the shoulders and a little toward +y, stored as `observation.images.image_front`. `camera_wrist_right` is `observation.images.image_wrist`. 256×256 RGB. `frontcam` is unused. The key name `image_front` is the SmolVLA name; the image is `tablecam`. The old overhead pose is not this camera.
- Instruction, every episode: `place the peg in the hole`.
- Grasp is friction only. `grasp_right_peg` stays disabled, same rule as the throw weld. The approach has to be fully open.
- Success: peg center stays inside the socket opening for 0.5 s. Log the failure as `never_grasped`, `dropped`, `missed`, or `timeout`.
- A run counts only with an `eval_summary.json` or, for the expert, a summary json. A loss number is not a result.
- fp32 only. Nothing written to `C:`. First line of every shell:

```powershell
$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"
```

- At most two demo collectors, and never while a GPU job is running.

## Steps

### Step 0 — scene and start pose

Load the copied scene, reset to key `home`, and confirm both arms sit in that pose over the table. Render `tablecam` and `camera_wrist_right` once and save the frames.

Done when: a reset lands on key 0, both frames exist, and the socket can be moved because the copy gave it a free joint.

Blocks everything else. One agent.

### Step 1 — expert

A scripted expert, shaped like `openarm_vla/expert/throw_expert.py`: an open-loop sequence planned from the true peg and socket positions, then replayed as joint targets. Phases are hover, open, descend, close, lift, carry, lower through the hole center, open, retreat. No throw.

It may read body positions. Insertion is a straight lower onto the socket center, using the same IK helper (`openarm_vla/expert/ik.py`). Tune the approach height and the lower speed until the peg does not jam on the octagon walls.

Done when: at least 20 random but legal layouts, from `peg_socket/findings/expert/`, and the summary json says how many inserted. Target before any data collection: 18/20 or better. The two failures get a named cause.

Depends on step 0. CPU only.

### Step 2 — load the base VLA on this scene

Wire the two cameras into `SmolVLAAdapter` and step the sim for a few chunks with `lerobot/smolvla_base`. Confirm the observation keys, the 15-D state, and the 8-D action. Base actions are meaningless until fine-tuned (the released head is a 6-D SO-100). This step checks that the robot accepts them without a crash.

Done when: one episode runs, frames and action shapes are recorded, and `peg_socket/findings/smoke/` says whether it crashed.

Depends on step 0. Short GPU job. Does not wait for the expert.

### Step 3 — demonstrations

Only after the expert hits its 18/20 bar. Randomize peg and socket on the table, then keep the draw only when all of these hold:

- both objects are on the tabletop, clear of the pedestal and of the home gripper footprint
- the peg is far enough from the socket that the fingers can open around it
- both objects project inside `tablecam` with margin, not on the frame edge

Record the bounds in the report so the next agent uses the same draw. Success episodes only. Start at 50 train and 20 val, no shared seeds, matching `data/datasets/train_head`. `--out` must not already exist.

Done when: `dataset_stats` has been run and the report lists episode counts, the bounds, and where the rejected draws went.

Depends on step 1. CPU collectors. Must not run during step 4, 5, or 6.

### Step 4 — zero-shot

Roll out `lerobot/smolvla_base` on 20 legal random layouts with the fixed sentence. Expect failure. The point is a recorded baseline: grasp count and how close the hand came to the peg.

Done when: `eval_summary.json` is in `peg_socket/findings/zeroshot/`.

Depends on step 2. GPU. Does not need demonstrations. Must not overlap step 3.

### Step 5 — can it see the peg

On the validation layouts, ask the frozen SmolVLM tower for the peg's pixel in `tablecam`, and train nothing. Score centimeters on the table, and whether the point lands on the peg. The same measurement as `scripts/vlm_point.py` and `findings/vision_xy/REPORT.md`.

Gate: median error under 1 cm, and the point on the peg in most frames. The hole only allows about 6 mm, so 1 cm is already a loose gate. If the gate fails, stop. Do not start step 6. Write what the replies actually were. On the balls, every reply was `10, 10` or `100, 100`.

Done when: `peg_socket/findings/localize/REPORT.md` states pass or fail against that gate.

Depends on step 0 for renders. Uses the step 3 validation layouts if they exist; otherwise its own 20 resets. GPU. After step 4, and not during collection.

### Step 6 — fine-tune the action expert

Only if step 5 passes. Fine-tune SmolVLA on the step 3 demos with vision and the language model frozen (`freeze_vision_encoder` and `train_expert_only`), fp32, the same recipe as `scripts/train_smolvla.py`. Output dir must not exist.

Closed-loop eval on the 20 held-out layouts. The bar the user set is 10% inserted, meaning 2 of 20, with `eval_summary.json`. Also report grasp rate and the distance from the peg to the hole at the closest point. A low flow-matching loss is not success.

If step 5 failed and someone still wants to know what action-only imitation does, that is a separate negative run, named as such, and it does not count as this step.

Depends on steps 3 and 5. One GPU job. No collectors alongside it.

## Subagents

Each agent writes its own report under `peg_socket/findings/` before it is finished. Two agents do not edit the same file. After step 0, nobody edits the scene copy except to fix a bug that blocks them, and they say so in the report.

| Agent | Does | Must not |
|---|---|---|
| Scene | Step 0. Copy the XML, add the socket free joint, prove key 0 and the two cameras. | Write the expert, load a checkpoint, collect demos. |
| Expert | Step 1, then step 3. Own the expert and the collect script. | Load SmolVLA, train, edit the camera wiring. |
| Smoke | Step 2, then step 4. Own the load and the zero-shot eval. | Retune the expert, collect demos, fine-tune. |
| Localize | Step 5. Own the pointing measurement. | Fine-tune, change the expert, collect the training set. |
| Train | Step 6 only, and only after the step 5 report says pass. | Collect demos while training, unfreeze the language model, claim success without `eval_summary.json`. |

## What can run together

```
step 0
   |
   +-- step 1 (CPU) --------+-- step 3 (CPU collectors)
   |                        |
   +-- step 2 (short GPU) --+-- step 4 (GPU)
                            |
                            +-- step 5 (GPU) --+-- step 6 (GPU), only if step 5 passed
```

- Step 1 and step 2 start together once step 0 is done.
- Step 4 waits for step 2, and does not wait for the expert.
- Step 3 waits for the expert's 18/20 bar.
- Step 3 and any GPU job (steps 4, 5, 6) do not overlap. Windows commit limit is about 38 GB, and a collector plus a GPU job is how the throw runs ran out of memory.
- Step 1 may overlap step 4. The expert is CPU. Zero-shot is one forward pass at a time.
- Step 5 waits until the GPU is free. It can use step 3's validation seeds if collection has finished; otherwise it renders its own 20.
- Step 6 is last.

## Also consider

- The expert can succeed while the policy cannot. True-state insertion at 6 mm does not say the cameras are good enough. Step 5 is the check.
- Copy the scene. Adding a free joint in `v2/pedestal/peg_socket_scene.xml` would change a file this repo treats as fixed.
- Keep the grasp weld off. A weld would make demos the policy cannot imitate.
- Write the randomization bounds before collecting, and save the rejected count. Otherwise a later agent cannot tell a hard layout from an illegal one.
- Do not point step 4 or step 6 at `checkpoints/` from the throw runs or at `data/datasets/train_head`. Those cameras and objects are different. `tablecam` is not `headcam`.
- Judge step 6 by insertions on new layouts. The throw fine-tunes drove the loss down and still missed the ball by centimeters.
- If step 5 fails, the useful next experiment is a different localizer, not more demonstrations and not an action-only fine-tune.
