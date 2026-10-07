# What the arm can do

The best checkpoint is `best/`. `view.py` loads it. The training record is [findings/so_xy_cont](../findings/so_xy_cont/REPORT.md).

On the held-out seeds 20000–20009 it stacks the red cube on the green cube 5 times out of 10. It grasps on 9 of 10. Three of those grasps are drops. The closest the red cube comes to sitting on the green one is 0.4 cm on the best episode and 2.1 cm on average.

The sentence it was trained on is `stack the red cube on the green cube`. The viewer will accept another sentence, but no other sentence was in the training set.

What it sees each time it plans:

- Top camera and wrist camera.
- Its six joint angles, in degrees. Pitch is negated.
- The sentence.
- The true red xy and green xy, in meters, z-scored and written into state dimensions 6:10. The mean is `0.00715, -0.21075, -0.00502, -0.21060` and the std is `0.07310, 0.03570, 0.06953, 0.03846`, from the train labels. The simulator fills those numbers. The policy does not measure them from the image.

It replans every 10 steps at 30 Hz, with physics paused while the chunk is computed. After a drop, the next chunk is computed from where the cubes are now, so it can try again.

The vision encoder and the language model are the pretrained SmolVLA weights. They were not updated. The action expert and `state_proj` were trained, in fp32, on 50 scripted successes. How that was measured is in [findings/so_stack](../findings/so_stack/REPORT.md), [findings/so_see](../findings/so_see/REPORT.md), [findings/so_xy](../findings/so_xy/REPORT.md), and [findings/so_xy_cont](../findings/so_xy_cont/REPORT.md).

Open it with:

```powershell
$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"
uv run python trs_so_arm100/view.py
```

`r` homes the arm and draws a new layout in the same floor patch as the demos: x in (−0.12, 0.12), y in (−0.28, −0.15), cubes at least 8 cm apart. Enter replaces the sentence. `q` quits.
