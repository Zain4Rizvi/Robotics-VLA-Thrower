# can it see the peg

Status: blocked

## Result

The 20 validation layouts replay, and the frozen SmolVLM2 tower was not asked. Loading `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` in fp32 ran out of GPU memory while `scripts/vlm_chat.py` (pid 8256) already held the 6 GB card. Median centimeters, frames on the peg, and parsed replies were not measured, so this is not a pass and not a gate failure. Step 6 may not start.

## Process

From the repo root, with `HF_HOME`, `UV_CACHE_DIR`, `TEMP`, `TMP`, `TORCH_HOME`, and `XDG_CACHE_HOME` on `Z:`, and `MUJOCO_GL=glfw`:

```
.venv\Scripts\python.exe -u peg_socket\localize_peg.py
```

`peg_socket/collect_peg_demos.py` and `scripts/train_smolvla.py` were not running. The model id is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. The script puts it in eval, turns `requires_grad` off, and does not build an optimizer. The process died inside `model.to("cuda")` before any peg question, so no weight was updated.

The 20 val seeds, in order, are 5002, 5008, 5009, 5013, 5020, 5030, 5035, 5039, 5041, 5045, 5055, 5057, 5066, 5067, 5069, 5071, 5079, 5085, 5092, 5108. Each one is `eval_peg_expert.draw` / `prepare` with instruction `place the peg in the hole`. The redrawn peg and socket xy match `data/datasets/peg_val/openarm_seeds.json`. After settle, the peg xy stays within 0.093 mm of that row. `grasp_right_peg` stays off. The frame is one 256×256 `tablecam` image. `frontcam` is unused.

Centimeters, once a reply exists: cast the `tablecam` ray through the parsed pixel and intersect the horizontal plane at that peg center's z. The error is the horizontal distance from the hit to the peg center, in centimeters. The median is over all 20 frames. An unparsed reply stays in the median as 1e6 cm, which is greater than 1 cm. A point is on the peg only when it falls inside the projected disk of the 1.4 cm peg radius (about 4.88 px here), the same rule as `on_ball`. The true pixel's own ray misses by 0.0000 cm on every frame, so the replay and the ray are the layouts that would have been scored.

## Numbers

| median cm | n_on_peg | n_parsed |
|---|---|---|
| not measured | not measured | not measured |

No replies. `points.json` was not written.

## Failures other agents should know

The card had 2234 MiB free. The fp32 load allocated 1.73 GiB and then failed asking for another 182 MiB (`torch.OutOfMemoryError` in `localize_stderr.txt`). Pid 8256 is `C:\Python312\python.exe -u scripts\vlm_chat.py`, the same SmolVLM2 checkpoint, resident on the GPU. It was already running. It was not killed.

Do not retry this step until that process has released the card and a fresh fp32 load fits. Do not switch to bf16 or fp16. Do not start a demo collector to work around it. The val seeds replay; do not render a replacement set of 20. There is no evidence yet about constant pixels on the peg, because the tower was not asked.

## Sources

- `peg_socket/localize_peg.py`
- `peg_socket/findings/localize/localize_stdout.txt`
- `peg_socket/findings/localize/localize_stderr.txt`
