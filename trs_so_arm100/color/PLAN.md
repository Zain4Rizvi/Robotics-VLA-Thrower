# Color words, then a stack

`best/` stacks the red cube on the green cube 5 of 10 times when it is told the true cube positions. It has only ever seen that one sentence. Asking it to stack the green cube on the red one does nothing useful: the language model was frozen, every demo used the same words, and state dimensions 6:10 are always red xy then green xy.

This sitting teaches the sentence to pick which cube moves. Vision stays frozen. The language model stays frozen. The true positions stay in the state. The color is not smuggled in by reordering those positions to mean "source, then target."

Three steps. Each one finishes its report, graphs, and videos before the next step starts. A step that scores 0 stacks is still written down. Do not delete or overwrite `best/`. Do not unfreeze SigLIP. Do not start a predicted-xy run.

## What "done" means for this sitting

The checkpoint to reach is step 3: a cube color that never appeared in training, named in the sentence, stacked on a held-out layout. Step 1 only has to show that the two red/green sentences do different things. Step 2 has to show that the trained color words work. Step 3 is the generalization check, and it is an eval of the step 2 weights unless that eval is 0 stacks and step 2 itself was stacking.

## Machine

GTX 1660, fp32. Vision frozen. Language frozen. Action expert and `state_proj` train. Batch 2 if the desktop is already on the card (GPU memory used ≥ 800 MB), otherwise 8. One GPU job. Nothing on `C:`. First line of every new shell:

```powershell
$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"
```

Long job: `Start-Process .venv\Scripts\python.exe ... -RedirectStandardOutput/-Error ... -WindowStyle Hidden`. Abort with `Stop-Process`, including dataloader workers.

About two hours per 2000-step train at batch 2 if validation is every 500 steps, then about half an hour to roll out. The whole sitting is a few hours. The report for a step wins over starting the next train.

## Already true

- Layout, cameras, joint signs, control rate, and the xy mean/std are in `trs_so_arm100/view.py`. Pitch is negated. 30 Hz, replan every 10. Physics stays paused while a chunk is computed.
- Floor patch: x in (−0.12, 0.12), y in (−0.28, −0.15), cubes at least 8 cm apart.
- xy mean `(red x, red y, green x, green y)` = `0.00715, -0.21075, -0.00502, -0.21060`. Std = `0.07310, 0.03570, 0.06953, 0.03846`. Keep using it. The floor patch did not change.
- State dimensions 0:6 are the six joint angles in degrees, normalized the way the `best/` checkpoint already normalizes them. Dimensions 6:8 are the `red_box` body xy. Dimensions 8:10 are the `green_box` body xy. Dimensions 10:32 stay zero. Those body names stay, even after a geom is recolored. Do not sort the slots into source-then-target.
- Success: the source cube is on the target cube, in contact, for 0.5 s (15 steps at 30 Hz). The target cube is still on the floor.
- Sentence shape: `stack the {source} cube on the {target} cube`.
- Init weights for step 1: `trs_so_arm100/best/pretrained_model`.
- How the last success was measured: `findings/so_xy_cont/REPORT.md`.

## Expert

Rebuild a short scripted expert in this folder. It may read cube poses. Grasp the source, lift, set it on the target, open. Same layout draw as `view.py`. Gate: 18 of 20 successes on fresh seeds before any dataset is written. Below that, fix the expert. Do not train.

Recolor by setting the cube geom `rgba` in Python at reset. Do not change masses, friction, or the floor. The floor geom stays `groundplane`.

## Step 1 — green on red

Cubes stay red and green, on the bodies `red_box` and `green_box`.

Two sentences, both in the training set:

- `stack the red cube on the green cube`
- `stack the green cube on the red cube`

Collect successes only. 40 train episodes and 16 val episodes of each sentence. Val seeds disjoint from train. One dataset, both sentences. Keys: `observation.images.camera1`, `observation.images.camera2`, `observation.state` (6, float32, degrees), `action` (6, float32, degrees), plus the task string. Store the seed, the sentence, and which body is the source.

Train from `best/` for 2000 steps. Fresh Adam, new cosine, warmup 100, peak lr `1.0e-4`, decay over 2000. fp32. Vision frozen. Language frozen. Save at 1000 and 2000. Train loss every step. Val loss every 500, on the val set only, xy filled the same way. Output directory `trs_so_arm100/color/swap/checkpoints`. It must not already exist.

Roll out the init weights and each saved checkpoint on seeds 30000–30009, twice: once for each sentence. Do not fit on these seeds.

Write `trs_so_arm100/color/swap/REPORT.md` from `eval_summary.json`. Table per sentence: step, stacks/10, grasps/10, drops, closest. `loss.png` and `success.png` (both sentences on the success graph). Videos, top camera with the wrist inset: `videos/expert.mp4`, and for each sentence `videos/red_on_green_step0000.mp4`, `videos/red_on_green_step1000.mp4`, `videos/red_on_green_step2000.mp4`, and the same three for `green_on_red`. Step 0 is `best/` on this task. The green-on-red step 0 clip should show the failure this sitting is trying to fix.

Move on after the report exists. A drop in red-on-green stacks is a result, not a reason to delete `best/`.

## Step 2 — trained color words

Four names, used in the sentence and painted on the geoms:

| name | rgba |
|---|---|
| red | 0.8 0.15 0.1 1 |
| green | 0.1 0.7 0.15 1 |
| blue | 0.15 0.25 0.85 1 |
| yellow | 0.9 0.8 0.1 1 |

Each episode draws two distinct colors and an order. The bodies stay `red_box` and `green_box`; only the rgba changes. State slots stay body order, as above. The sentence is the only thing that says which color is the source.

80 train episodes, 32 val, no shared seeds, colors and orders mixed. Same keys. Same train recipe as step 1, initialized from the step 1 checkpoint at 2000 steps, not from `best/` and not from `lerobot/smolvla_base`. Output directory `trs_so_arm100/color/palette/checkpoints`.

Roll out seeds 31000–31009. On each seed, sample the pair and the order from a fixed rng so the checkpoint comparison is the same scenes. Record the colors in `eval_summary.json`.

Write `trs_so_arm100/color/palette/REPORT.md`, `loss.png`, `success.png`. Videos: `videos/expert.mp4`, `videos/step0000.mp4`, `videos/step1000.mp4`, `videos/step2000.mp4`. Step 0 is the step 1 weights on these recolored scenes.

## Step 3 — a color the training set never said

No new demos unless step 2 stacked 0 of 10. In that case, stop after the step 2 report. Do not run step 3.

Otherwise take the step 2 checkpoint at 2000 and roll it out. Do not train. Colors are only `orange` (0.9 0.4 0.05 1) and `purple` (0.55 0.15 0.7 1). Same body-order state. Seeds 32000–32009, both orders across the 10 seeds. The words orange and purple must not appear in the step 1 or step 2 datasets.

Write `trs_so_arm100/color/heldout/REPORT.md` and `success.png` is not required; there is one checkpoint. One table: stacks/10, grasps/10, drops, closest. Videos: `videos/orange_on_purple.mp4` and `videos/purple_on_orange.mp4`.

If those 10 are all misses, write that the trained color words work only for colors named in the demos, and stop. Do not fine-tune SigLIP in this sitting.

## Stop

- Expert under 18/20: stop before the dataset.
- Dims 6:10 do not reach `state_proj`: stop. Write it in that step's report.
- A crash that fails the same way on one retry: stop. Write the traceback in the report.
- Step 2 stacked 0 of 10: stop after its report.
- Step 3 written, whatever the count.

Do not stop step 1 because green-on-red scored 0 and red-on-green still works. Write it and continue to step 2 only if green-on-red scored at least 1 stack. If it scored 0, stop after the step 1 report. The palette run is not useful until the two red/green sentences can be told apart.
