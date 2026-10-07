# Can the frozen sentence tell red from green?

`best/` stacks the red cube on the green cube when it is given the true cube positions. `color/swap` then trained the action expert on both sentences, with the language model frozen and the state slots fixed as red xy then green xy. Green on red never grasped. Red on green fell from 2 of 10 stacks to 0. The record is [color/swap/REPORT.md](../color/swap/REPORT.md). The palette and the held-out colors were not started, and this sitting does not start them.

The sentence is the only difference between those two tasks. It is turned into a block of numbers by 16 frozen text layers. The action expert only sees that block. This sitting asks whether those numbers already say which cube should move. The arm is not trained until that answer is written down.

Two steps. The probe finishes its report and its pictures before any training starts. The probe result picks exactly one of the two trains. Do not delete or overwrite `best/`. Do not unfreeze SigLIP. Do not reorder state into source-then-target. Do not start a predicted-xy run.

## What "done" means for this sitting

The probe report exists, with the three plots and the frame stills. Then exactly one train report exists, chosen by the probe gate below. A train that stacks 0 of 10 is still written down. The other train is not run.

## Machine

RTX 3080, 10 GB. Ampere supports bf16. The load path still casts the policy to fp32 (`use_amp=false` and `.float()`). Leave that cast. Dataset arrays stay float32. One GPU job. Nothing on `C:`. First line of every new shell:

```powershell
$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"
```

Long job: `Start-Process .venv\Scripts\python.exe ... -RedirectStandardOutput/-Error ... -WindowStyle Hidden`. Abort with `Stop-Process`, including dataloader workers. The probe is a forward pass. A 2000-step train is the long job. The report for a step wins over starting the train.

Pass `--batch-size` from free VRAM. The old rule (batch 2 when used GPU memory is ≥ 800 MB) was the 6 GB GTX 1660, and `color/train.py` still applies it. Do not use that default. Expert-only training starts at batch 8. The text-layer train starts at batch 4. If the first steps run out of memory, drop the batch and retry once. A second failure of the same kind stops the sitting.

## Already true

- Layout, cameras, joint signs, control rate, and the xy mean/std are in `trs_so_arm100/view.py`. Pitch is negated. 30 Hz, replan every 10. Physics stays paused while a chunk is computed.
- Floor patch: x in (−0.12, 0.12), y in (−0.28, −0.15), cubes at least 8 cm apart.
- xy mean `(red x, red y, green x, green y)` = `0.00715, -0.21075, -0.00502, -0.21060`. Std = `0.07310, 0.03570, 0.06953, 0.03846`. Keep using it.
- State dimensions 0:6 are the six joint angles in degrees, normalized the way `best/` already normalizes them. Dimensions 6:8 are the `red_box` body xy. Dimensions 8:10 are the `green_box` body xy. Dimensions 10:32 stay zero, except in the train that writes the probe scores into 10:12. Body names stay. Do not sort the slots into source-then-target.
- Sentences: `stack the red cube on the green cube` and `stack the green cube on the red cube`.
- Success: the source cube is on the target cube, in contact, for 0.5 s (15 steps at 30 Hz). The target cube is still on the floor.
- Init weights: `trs_so_arm100/best/pretrained_model`. The `color/swap` checkpoints already forgot the stack. Do not init from them.
- Seeds and episodes already on disk: `trs_so_arm100/color/swap/datasets/train/stack_seeds.json` and `trs_so_arm100/color/swap/datasets/val/stack_seeds.json`. Do not collect another dataset.
- How the frozen prefix is built: [findings/smolvla_architecture.md](../../findings/smolvla_architecture.md). Cameras, then word tokens, then one state token. The word tokens can see the cameras. They cannot see the state token.

## The note

Load `best/` in fp32, eval mode, everything frozen. For each unique seed in the swap train file and the swap val file, reset once with `view.py`'s layout draw. Capture the top camera, the wrist camera, the joint angles, and the true red xy and green xy. Run the prefix twice on that same capture, once with each sentence.

The note is the mean of the word-token hidden states after the 16th text layer. Width 960. Leave out the 64 front-camera tokens, the 64 wrist tokens, and the state token. A mean over the whole prefix lets the cameras drown the two color words. If the word span is empty or the width is not 960, stop and write the shapes in the probe report.

Standardize the notes with the train mean and std. Fit an L2 logistic regression on the train notes (`C=1.0`). The label is which body the sentence names as the source. Score it on the val notes. Then shuffle the train labels with seed 0, refit, and score that second classifier on the true val labels.

Also store, for every matched pair, the cosine distance between the two notes from the same frame.

Do not fit on seeds 30000–30009. Those are the rollout seeds.

## What the report has to show

Every step writes its `REPORT.md` before the next step starts. Counts and accuracies come from the json next to that report. The pictures are part of the result. A reader who only opens the report should see the comparison without opening a log.

Logs stay in the step folder (`stdout`, `stderr`, and the json). The report quotes them. It does not replace them.

## Step 1 — the probe

No arm rollout. No training. Output directory `trs_so_arm100/sentence/probe/`.

Write `probe.json`: train accuracy, val accuracy, shuffled-val accuracy, number of train seeds, number of val seeds, and the cosine distances.

Plots:

- `accuracy.png` — three bars: train, val, shuffled val. A line at 0.5.
- `distance.png` — cosine distance between the two notes on the same frame, train and val separate.
- `notes.png` — two-component PCA fit on the train notes, val notes drawn and colored by sentence.

Stills: `frames/` with four val seeds. Each image is the top camera. Write both sentences on it and the classifier's call for each sentence. These frames show that the cubes did not move between the two notes.

`REPORT.md` leads with the three accuracies and which gate below fired. Embed the three plots and one still.

### Gate

- The note works: val accuracy ≥ 0.90 and shuffled-val accuracy ≤ 0.60. Run step 2. Do not run step 3.
- The note fails: val accuracy ≤ 0.65, or val accuracy is within 0.10 of the shuffled-val accuracy. Run step 3. Do not run step 2.
- Anything between those two: stop after the probe report. Do not train. Write that the note is only partly separated, so neither train is justified yet.

## Step 2 — the note works, so teach the arm a number

The frozen classifier already reads the sentence. This train asks whether the action expert can use that answer once it is a pair of numbers in the state, which is the same kind of input as the true xy that made stacking work.

Save the probe's standardization and classifier in `probe/`. During training and rollout, run the frozen prefix, apply that frozen classifier, and write its two logits into state dimensions 10:12. Dimensions 6:10 stay the body xy. Gradients stop at the logits. The text layers, SigLIP, the connector, and the classifier stay frozen. The action expert and `state_proj` train.

Same swap datasets. Fill xy the way `color/train.py` does. Fresh Adam, new cosine, warmup 100, peak lr `1.0e-4`, decay over 2000. fp32. Batch 8 to start. Save at 1000 and 2000. Train loss every step. Val loss every 500, on the val set only, xy and the two logits filled the same way. Output directory `trs_so_arm100/sentence/use_note/checkpoints`. It must not already exist.

Roll out seeds 30000–30009, both sentences, at steps 0, 1000, and 2000. Step 0 is `best/` with the logits already written in. Do not fit on these seeds.

Write `trs_so_arm100/sentence/use_note/REPORT.md` from `eval_summary.json`. Table per sentence: step, stacks/10, grasps/10, drops, closest. `loss.png` and `success.png` (both sentences on the success graph). Videos, top camera with the wrist inset: `videos/red_on_green_step0000.mp4`, `videos/red_on_green_step1000.mp4`, `videos/red_on_green_step2000.mp4`, and the same three for `green_on_red`. One still of a success if any exist, otherwise the closest miss, for each sentence at step 2000: `frames/red_on_green.png` and `frames/green_on_red.png`.

Stop after this report. Do not unfreeze the text layers because the arm still missed.

## Step 3 — the note fails, so let the text layers learn

Run this only when the probe gate says the note fails. Init is `best/`, not the swap checkpoint and not step 2.

`train_expert_only=false` so the text layers can learn. `freeze_vision_encoder=true`. Also freeze the connector. The library freezes `lm_head`, the final text norm, and the last text layer (`text_model.layers.15`). Leave that freeze. Before the first optimizer step, print the trainable count and stop if any vision-tower or connector parameter is trainable, or if every text layer is frozen.

Two learning rates, one cosine, warmup 100, decay over 2000. Text layers `1.0e-5`. Action expert and `state_proj` `1.0e-4`. fp32. Batch 4 to start. State dims 10:32 stay zero. Save at 1000 and 2000. Train loss every step. Val loss every 500. Output directory `trs_so_arm100/sentence/text_layers/checkpoints`. It must not already exist.

Same rollout as step 2, on seeds 30000–30009, both sentences. Step 0 is `best/` with dims 10:32 zero.

Write `trs_so_arm100/sentence/text_layers/REPORT.md` from `eval_summary.json`. Same tables, `loss.png`, `success.png`, and the six videos. Stills: `frames/red_on_green.png` and `frames/green_on_red.png` at step 2000, a success if any exist, otherwise the closest miss. The report names which text layers were trainable.

Stop after this report. Do not run step 2 afterwards.

## Stop

- The word-token span cannot be found, or its width is not 960: stop in the probe report.
- A crash that fails the same way on one retry: stop. Write the traceback in the report.
- The probe lands between the two gates: stop after the probe report.
- Step 2's report is written, or step 3's report is written. The other train stays unstarted.
- Do not continue to blue, yellow, orange, or purple.
- Do not fine-tune SigLIP.
- Do not train into `best/` or into `color/swap/checkpoints`.
