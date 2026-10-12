# Camera and sentence, no cube positions

Init is a copy of `trs_so_arm100/sentence/cont/run1/checkpoints/checkpoints/008000/pretrained_model`, stored at `trs_so_arm100/camera_text/init`. It is not `lerobot/smolvla_base`. Those weights already grasp and stack when red xy, green xy, and the probe logits are in the state (7/10 and 6/10 at step 8000). This run asks whether that expert can relearn both sentences from the cameras and the task string alone. The base-model camera stack never closed the spawn gap. SigLIP stays frozen: the frozen readout is about 1.6 cm, and training the tower made that worse. Text layers stay frozen. Early checkpoints can get worse because those weights expect xy. Step 0 is rolled out before any training step.

## State

Dimensions 0:6 are the six joint angles. Dimensions 6:32 stay zero at train time and at rollout, including xy and the probe logits. The swap `xy.npz` is not read during training. Closest at rollout is the live sim sit pose. The train step and every rollout chunk abort if any value in dimensions 6:32 is nonzero at `state_proj`.

## Budget

8000 steps, fresh Adam, new cosine, peak learning rate `1.0e-4`, warmup 100, decay over those 8000 steps, floor `2.5e-6`. fp32, `use_amp` false. Save every 2000 steps, and keep a copy of the init weights at step 0. Train loss every step. Val loss every 500 steps. After the first 1000 steps, slower than 45 minutes per 1000 steps cuts the run to 6000 and retargets the cosine.

Batch starts at 8. If the first steps leave at least 1 GB free, stop, delete the output directory, and try batch 16 once. On CUDA OOM, delete that output directory, halve the batch, and retry once. A second OOM stops the run.

## Rollout

Seeds 30000–30009, both sentences, disjoint from the swap train and val seeds. Steps 0, 2000, 4000, 6000, and 8000, dropping 8000 if the run was cut. One video per sentence per checkpoint, seed 30000. Stills at the last rolled-out step only.

Do not write into `best/`, `sentence/cont/run1/`, `sentence/use_note/`, `color/swap/`, `red_green_view.py`, `view.py`, or `playground/`. Do not point the web app at this checkpoint.
