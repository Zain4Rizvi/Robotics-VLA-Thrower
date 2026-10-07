# Both sentences, then a held-out color pair

`use_note` already showed the arm can use the sentence once it is two numbers in the state. At step 2000, red on green stacked 5 of 10 and green on red stacked 2 of 10. Both counts were still rising, and the learning rate was already at the floor (`2.5e-6`). The record is [use_note/REPORT.md](../use_note/REPORT.md).

Two phases. Phase 1 keeps those weights, restarts the cosine, and trains until green on red can reach 5 of 10 on the same seeds as red on green. Phase 2 starts only after that bar is met. It teaches four color words and leaves one pairing out of the training set: `stack the yellow cube on the blue cube`. A table that stops at 2 of 10 is still the result. Do not claim a count without that run's `eval_summary.json`.

Phase 1 is run 1, then run 2 only when the bar is still missed. Phase 2 is a new dataset, a new probe, and a new train, under `sentence/cont/colors/`. Do not delete or overwrite `best/` or `use_note/`. Do not unfreeze SigLIP. Do not unfreeze the text layers. Do not refit `sentence/probe/classifier.npz`. Do not train orange or purple. Do not put the yellow-on-blue sentence in the color training set or the color val set.

## What "done" means

Run 1's report exists, with the loss plot, the success plot, the videos, and the saved checkpoints. Then one of these is true:

- The phase-1 bar is met and the color reports exist: `colors/probe/REPORT.md`, `colors/train/REPORT.md`, and `colors/unseen_words/REPORT.md`.
- The phase-1 bar is met and the clock says not to start colors. Run 1's report, or run 2's if run 2 happened, names that branch.
- The phase-1 bar is still missed and run 2's report exists, or the clock says not to start run 2.

The last report states the red-on-green and green-on-red stack counts. If colors ran, it also states the yellow-on-blue count, the blue-on-yellow count, and the orange/purple counts.

## Clock

Spend at least 6 hours and stop by 8 hours 30 minutes, measured from the moment run 1's training process starts. The report for a run wins over starting the next one. Do not start a train you cannot roll out before the cap.

After the first 1000 steps, read the step rate from `train_log.csv` and the wall clock.

- Faster than 45 minutes per 1000 steps: run 1 stays 8000 steps.
- Slower than 45 minutes per 1000 steps: cut run 1 to 6000 steps so the rollout still fits. Write the cut in the report.
- After the phase-1 report that decides the next branch, start that next train only if the time below still remains. Otherwise stop and write which branch would have fired.

Run 2 needs at least 2 hours 30 minutes. Colors start when at least 1 hour remains. Under 1 hour, do not start them. If at least 5 hours remain when colors start, the color train is 4000 steps. If at least 3 and under 5 remain, it is 2000 steps. If at least 1 and under 3 remain, it is 1000 steps. If the measured step rate says that budget cannot be rolled out before the cap, cut to the longest multiple of 1000 that still finishes. If 1000 itself will not finish, do not start. Write the cut in the color report.

## Machine

RTX 3080, 10 GB. Ampere supports bf16. The load path still casts the policy to fp32 (`use_amp=false` and `.float()`). Leave that cast. Dataset arrays stay float32. One GPU job. Collection is CPU and finishes before the probe forward pass.

`Z:` was not mounted the last time this machine trained. `Test-Path Z:\` before the first shell. If it is true, use the `AGENTS.md` prelude. If it is false, do not point `HF_HOME` at `Z:`. The SmolVLM2 snapshot is already in the user cache, and a missing `Z:` kills the load with `WinError 3`. Use this prelude instead, and create `D:\tmp`, `D:\uv_cache`, `D:\torch`, and `D:\xdg` if they are absent:

```powershell
$env:HF_HOME="C:\Users\Zain\.cache\huggingface"; $env:HUGGINGFACE_HUB_CACHE="C:\Users\Zain\.cache\huggingface\hub"; $env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; $env:UV_CACHE_DIR="D:\uv_cache"; $env:TEMP="D:\tmp"; $env:TMP="D:\tmp"; $env:TORCH_HOME="D:\torch"; $env:XDG_CACHE_HOME="D:\xdg"; $env:MUJOCO_GL="glfw"
```

Offline mode is there so the load does not try to download. If a required file is actually missing from that cache, stop and write the missing path. Do not switch the cache onto `C:` mid-run.

Long job: `Start-Process .venv\Scripts\python.exe ... -RedirectStandardOutput/-Error ... -WindowStyle Hidden -PassThru -Wait`. Logs go in that run's folder (`stdout.txt`, `stderr.txt`). Abort with `Stop-Process`, including dataloader workers. Killing the terminal does not stop a hidden process.

Batch 8 is the start. The `use_note` train held about 5.8 GB at batch 8. If the first steps run out of memory, delete that run's output directory, drop to batch 4, and retry once. A second failure of the same kind stops the sitting. Write the traceback in the report.

`color/train.py` still contains `assert_expert_only`, which refuses any schedule whose decay is not 2000. Do not weaken that function. A script in this folder checks its own schedule.

## Already true

- Layout, cameras, joint signs, control rate, and the xy mean/std are in `trs_so_arm100/view.py`. Pitch is negated. 30 Hz, replan every 10. Physics stays paused while a chunk is computed.
- Floor patch: x in (−0.12, 0.12), y in (−0.28, −0.15), cubes at least 8 cm apart.
- xy mean `(red x, red y, green x, green y)` = `0.00715, -0.21075, -0.00502, -0.21060`. Std = `0.07310, 0.03570, 0.06953, 0.03846`. Keep using it. The two slots stay `red_box` then `green_box` even after a geom is recolored.
- Phase 1 state: dimensions 0:6 are the six joint angles in degrees, normalized the way `best/` already normalizes them. Dimensions 6:10 are `red_box` xy then `green_box` xy, z-scored with that mean and std. Dimensions 10:12 are the phase-1 probe's two logits. Dimensions 12:32 stay zero. Body names stay. Do not sort the slots into source-then-target.
- The phase-1 logits are the frozen probe classifier in `sentence/probe/classifier.npz`, applied by `sentence/note.py`. Class 0 is `red_box`, class 1 is `green_box`. The two stored numbers are `(-s/2, s/2)`, where `s` is the L2 logistic decision function. Their difference is the log-odds of green as the source. Gradients stop at the logits. Do not refit that classifier. Reloading it on `probe/notes.npz` must still score val accuracy 1.0 before phase-1 training starts.
- Phase 1 sentences: `stack the red cube on the green cube` and `stack the green cube on the red cube`. The swap train file has 40 episodes of each. The val file has 16 of each.
- Success: the source cube is on the target cube, in contact, for 0.5 s (15 steps at 30 Hz). The target cube is still on the floor.
- Init weights for run 1: `trs_so_arm100/sentence/use_note/checkpoints/checkpoints/002000/pretrained_model`. Fresh Adam. Do not resume the old optimizer. The cosine decay floor stays the value already in that checkpoint (`2.5e-6`).
- Phase 1 seeds and episodes already on disk: `trs_so_arm100/color/swap/datasets/train` and `.../val`, including `xy.npz` and `stack_seeds.json`. Phase 1 does not collect another dataset.
- Phase 1 rollout seeds: 30000–30009, both sentences. Do not train on them. They are disjoint from the swap train and val seeds.
- How the frozen prefix is built: [findings/smolvla_architecture.md](../../../findings/smolvla_architecture.md). The word tokens can see the cameras. They cannot see the state token. The note code in `sentence/note.py` is the one the 2000-step run used. Reuse that note for both probes.
- Recolor is geom `rgba` only, in Python at reset. Do not change masses, friction, or the floor. The floor geom stays `groundplane`. The rgba table in `trs_so_arm100/color/expert.py` is the one to use.

## What the report has to show

Every run writes its `REPORT.md` before the next run starts. Counts come from that run's `eval_summary.json`. The pictures are part of the result.

Logs stay in the run folder (`stdout.txt`, `stderr.txt`, `checkpoints/train_log.csv`, `checkpoints/val_log.csv`, and the json). The report quotes the last-step stack line. It does not replace the logs.

Plots, both written from the csv and the json, not from memory:

- `loss.png` — train loss every step, val loss at every val check.
- `success.png` — stacks out of 10, one line per sentence that was rolled out.

Videos, top camera with the wrist inset, one clip per rolled-out sentence per rolled-out checkpoint. Pad the step number. Do not drop a checkpoint from the table because its video is a miss.

Saved weights: every `save_freq` checkpoint's `pretrained_model/model.safetensors` stays. Pruning old `training_state/` folders is fine. Deleting a `pretrained_model` is not.

## Run 1 — restart the cosine

Same wiring as `use_note`: frozen prefix, frozen phase-1 classifier, logits in dims 10:12, true xy in dims 6:10, action expert and `state_proj` train. Vision, text layers, and the connector stay frozen. Print the trainable count before the first optimizer step and stop if any vision-tower, connector, or text-layer parameter is trainable.

8000 steps, unless the clock section cuts that to 6000. Fresh Adam, new cosine, warmup 100, peak lr `1.0e-4`, decay over the step budget of this run. fp32. Batch 8. Save every 2000. Train loss every step. Val loss every 1000, on the val set only, xy and the two logits filled the same way. Output directory `trs_so_arm100/sentence/cont/run1/checkpoints`. It must not already exist.

Roll out seeds 30000–30009, both sentences, at steps 0, 2000, 4000, 6000, and 8000 (drop 8000 if the run was cut to 6000). Step 0 is the `use_note` step-2000 weights with the logits already written in. The earlier measurement of those same weights was 5 of 10 and 2 of 10. This step 0 is a new rollout. Noise makes the two draws differ. The report quotes both, and the table in this report is this rollout.

Phase 1 videos use seed 30000: `videos/red_on_green_step0000.mp4` at every saved step, and the same names for `green_on_red`. Stills at the last rolled-out step only: `frames/red_on_green.png` and `frames/green_on_red.png`. A success if that sentence stacked on any of the ten seeds, otherwise the closest miss.

Write `trs_so_arm100/sentence/cont/run1/REPORT.md` from `eval_summary.json`. Table per sentence: step, stacks/10, grasps/10, drops, closest. The report's first paragraph is the two stack counts at the last step, and whether each one is at least 5.

## Gate after phase 1

Read the counts from the latest phase-1 `eval_summary.json`. Let `R` be red-on-green stacks and `G` be green-on-red stacks at the last step. Let `G_prev` be green-on-red stacks at the checkpoint before that.

The phase-1 bar is `R >= 5` and `G >= 5`.

- The bar is met on run 1. Do not start run 2. Extra red/green steps would only memorize those two sentences. Start colors if the clock still allows them. Init colors from the earliest run-1 checkpoint where both counts are at least 5. Write that step in the color report.
- The bar is met only after run 2. Start colors if the clock still allows them. Init from the earliest run-2 checkpoint where both counts are at least 5.
- `G < 5` and `G > G_prev`, and run 2 has not been done. Green on red is still rising. Run 2 is the same phase-1 recipe for 6000 more steps, init from run 1's last checkpoint, new cosine, output `trs_so_arm100/sentence/cont/run2/checkpoints`. Save every 2000. Val every 1000. Then apply this gate again.
- `G < 5` and `G <= G_prev`, and run 2 has not been done. The equal-weight cosine has flattened short of the bar. Run 2 upweights green-on-red frames by 2 in the training loss only. Val stays unweighted. Init from the run 1 checkpoint with the highest `G` among those with `R >= 3`. If none have `R >= 3`, init from the last checkpoint. 4000 steps, new cosine, batch 8, same phase-1 state wiring. Output `trs_so_arm100/sentence/cont/run2/checkpoints`. The report names the weight and the init step. Then apply this gate again.
- The bar is still missed after run 2. Do not start colors. The run 2 report says the color phase was not started because green on red, or red on green, was still under 5.

If the clock does not allow the next train, the report that was just written says which branch would have fired and that it was not started.

Run 2's rollout is the same seeds and both sentences, at step 0 and every saved checkpoint. Step 0 is the init weights, so the table connects to run 1. Same plots, videos, and stills. Write `trs_so_arm100/sentence/cont/run2/REPORT.md`.

## Colors — a pairing the training set never showed

This phase exists to test one claim: the arm can stack yellow on blue after seeing yellow, seeing blue, and seeing stacking, without seeing that pairing. It is not a test of colors that never appear in any sentence. Orange and purple are a separate rollout at the end, with no training.

Folders, and nothing color-related is written outside them:

- `trs_so_arm100/sentence/cont/colors/datasets/train` and `.../val`
- `trs_so_arm100/sentence/cont/colors/probe/`
- `trs_so_arm100/sentence/cont/colors/train/`
- `trs_so_arm100/sentence/cont/colors/unseen_words/`

Do not write into `run1/`, `run2/`, `use_note/`, `best/`, or `color/swap/`.

### Sentences

Four names, painted with the rgba table in `color/expert.py`: red, green, blue, yellow. The bodies stay `red_box` and `green_box`. An episode paints one color on each body. The sentence names the colors. The sentence shape stays `stack the {source} cube on the {target} cube`.

Eleven training pairs. Each of the four colors is a source in at least one pair and a target in at least one pair:

- red on green, red on blue, red on yellow
- green on red, green on blue, green on yellow
- blue on red, blue on green, blue on yellow
- yellow on red, yellow on green

Held out of train and val: yellow on blue. The reverse, blue on yellow, stays in the training set. That is the control with the same two colors and the opposite order.

### Dataset

Script the expert in this phase's folder. It may read cube poses. The policy will not. Reuse the grasp, lift, and place from `color/expert.py`. Gate: 18 of 20 successes on fresh seeds drawn from the eleven pairs, seeds 60000–60019, before any dataset is written. Then 18 of 20 on yellow on blue alone, seeds 61000–61019. Those seeds are the expert check. They are not written into the dataset. Below either gate, fix the expert. Do not train.

Successes only. 10 train episodes of each of the eleven pairs. 4 val episodes of each. Val seeds disjoint from train. One train dataset, one val dataset. Attempt seeds start at 40000 for train and 50000 for val. Refuse to write an episode whose sentence is yellow on blue, or whose text contains orange or purple.

Same keys as `color/collect.py`: `observation.images.camera1`, `observation.images.camera2`, `observation.state` (6, float32, degrees), `action` (6, float32, degrees), plus the task string. `xy.npz` stores `red_box` xy then `green_box` xy in body order, paint ignored. `stack_seeds.json` stores the seed, the sentence, the source color, the target color, and which color was painted on which body.

### Probe

A new classifier. Do not load its weights from `sentence/probe/classifier.npz`, and do not overwrite that file.

The note is the same word-token mean as `sentence/note.py`, width 960. Capture it on the train episodes, the val episodes, and on yellow-on-blue frames from seeds 33000–33009. Those held-out frames are for the probe score only. They do not enter the dataset.

Standardize with the train-note mean and std. Fit two L2 logistic regressions, `C=1.0`, on the train notes only. One predicts the source color. One predicts the target color. Class order is red, green, blue, yellow. Score both on the val notes. Then score seeds 33000–33009: the fraction of those frames whose source argmax is yellow and whose target argmax is blue.

Write `colors/probe/classifier.npz`, `colors/probe/notes.npz`, and `colors/probe/probe.json` (train accuracy, val accuracy, and the held-out-sentence accuracy, for source and for target). `colors/probe/accuracy.png` is those bars, with a line at 0.9. Write `colors/probe/REPORT.md` before any color training.

Gate: val source accuracy ≥ 0.90, val target accuracy ≥ 0.90, and the held-out sentence ≥ 0.90 on both. If any of the three miss, stop after the probe report. Do not train. Do not unfreeze the text layers. The report says the frozen notes did not separate the color words.

### State for the color train

Dimensions 0:6 and 6:10 stay as in phase 1: joints, then `red_box` xy, then `green_box` xy. Dimensions 10:14 are the source-color logits, in the class order above. Dimensions 14:18 are the target-color logits, same order. Dimensions 18:32 stay zero. Gradients stop at the logits. Do not write which body is the source. A policy that is handed the source body does not have to read the color.

Vision, the text layers, the connector, and both color classifiers stay frozen. The action expert and `state_proj` train. Print the trainable count before the first optimizer step and stop if any vision-tower, connector, or text-layer parameter is trainable.

### Color train

Init from the phase-1 checkpoint named by the gate. Fresh Adam, new cosine, warmup 100, peak lr `1.0e-4`, decay over the step budget the clock chose (1000, 2000, or 4000). fp32. Batch 8. Save every 1000. Train loss every step. Val loss every 500, on the val set only. The val set has no yellow-on-blue episodes. Fill xy and the eight logits the same way as training. Output directory `trs_so_arm100/sentence/cont/colors/train/checkpoints`. It must not already exist.

The train script refuses to start if yellow on blue, orange, or purple appears in the train or val `stack_seeds.json`.

Roll out three sentences at step 0 and every saved checkpoint:

- Yellow on blue, seeds 33000–33009. This sentence is not in the training set. The layout for a seed paints the two bodies from that seed and keeps those poses for every checkpoint.
- Blue on yellow, the same seeds and the same layouts. This pair was trained. Same pixels, other sentence.
- Red on green, seeds 30000–30009, both bodies painted their own names. This is the regression against phase 1. Step 0 here is the phase-1 weights with the color logits in dims 10:18, so it is not the same measurement as run 1's red-on-green count. The report says that.

Videos, seed 33000 for the first two and seed 30000 for red on green: `videos/yellow_on_blue_step0000.mp4`, `videos/blue_on_yellow_step0000.mp4`, `videos/red_on_green_step0000.mp4`, and the same names at every saved step. `success.png` has one line per sentence. Stills at the last rolled-out step only: `frames/yellow_on_blue.png`, `frames/blue_on_yellow.png`, `frames/red_on_green.png`. A success if that sentence stacked on any of its ten seeds, otherwise the closest miss.

Write `trs_so_arm100/sentence/cont/colors/train/REPORT.md` from `eval_summary.json`. The json records, per sentence and per step, stacks/10, grasps/10, drops, and closest, plus the init checkpoint path and step. The first paragraph is the three last-step stack counts. Yellow on blue above 0 is the result this phase is looking for. 5 of 10 is not required. A 0 is written down. Do not start a second color train to chase a higher count.

### Unseen words

After the color-train report, if the clock still has room for a rollout and no new train, roll out the last color checkpoint. Do not train. Colors are only orange (`0.9 0.4 0.05 1`) and purple (`0.55 0.15 0.7 1`). Seeds 34000–34009, orange on purple for the first five and purple on orange for the last five. Those words must not appear in the color datasets or in the probe fit.

The four-way probe has no orange class and no purple class. Dimensions 10:18 are zero on this rollout. The sentence in the frozen prefix is the only color cue. The report says the color dims were zero.

Write `trs_so_arm100/sentence/cont/colors/unseen_words/REPORT.md` from its `eval_summary.json`. One table: sentence, stacks/10, grasps/10, drops, closest. Videos: `videos/orange_on_purple.mp4` and `videos/purple_on_orange.mp4`, seed 34000 and seed 34005. If those 10 are all misses, write that the color codes work for words the probe was fit on, and that orange and purple were not a training target.

## Stop

- The phase-1 classifier does not score val accuracy 1.0 on `probe/notes.npz`: stop before run 1. Write that in `run1/REPORT.md`.
- The color probe misses its gate: stop after `colors/probe/REPORT.md`. Do not start the color train.
- The expert is under 18/20 on either color gate: stop before the color dataset is written.
- A crash that fails the same way on one retry: stop. Write the traceback in the report.
- The clock is past 8 hours 30 minutes: finish the rollout that is already in progress if it is the one for a finished train, otherwise write the report from whatever checkpoints exist and stop. Do not start another train.
- Run 2's report is written and the phase-1 bar is still missed.
- The unseen-words report is written.
- The phase-1 bar is met, the clock does not allow colors, and that fact is in the phase-1 report.
- Do not fine-tune SigLIP.
- Do not unfreeze the text layers.
- Do not train into `best/`, `use_note/`, `color/swap/checkpoints`, or a previous run's checkpoint folder.
