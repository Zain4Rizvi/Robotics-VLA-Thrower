# Is a  SmolVLA fine-tune feasible?

Status: done

This reads the throw reports in `findings/` and the peg-socket record in `peg_socket/REPORT.md`. No new training run, no new rollout.

## Result

Training SmolVLA on either task is not feasible on this machine. The training loop runs and the flow-matching loss falls. Every closed-loop eval still misses the object by several centimeters, and that gap is the task.

The throw policies put 0 balls in the bin on the scored rollouts. The one recorded "in the bin" at the 50-demo checkpoint had the contact flag off. The only trained peg policy inserted 0/5 and grasped 0/5. A pinch is about 1 cm. The peg hole allows about 6 mm. The hands stopped 4–11 cm short.

Another fine-tune of this recipe, including peg step 6, would repeat a miss that is already on disk. The measurement that would change that is a localizer that beats the centimeter gate on held-out frames before the action head is trained. For the ball that gate is 5 cm and the best result is 9.0 cm. For the peg the gate is 1 cm and it was never scored.

## Throw

The scripted expert lands the ball in 87 of 100 episodes. It sees the true ball and bin positions. The policy sees cameras, joints, and the sentence.

| Run | Demos | In the bin | Grasps | Closest | Loss |
|---|---|---|---|---|---|
| [expert](../expert/REPORT.md) | script | 87/100 | 99/100 | lands | true positions |
| [overfit10](../overfit10/REPORT.md) | 10 | 0/20 | 3/20 | 5.8 cm mean | 2.98 → 0.027, no val loss |
| [stageA](../stageA/REPORT.md) | 50 | 1/20, contact flag off | 1/20 | 8.4 cm mean | val loss rose after step 2000 |
| [headcam](../headcam/REPORT.md) | 50 | 0/20 | 1/20 | 8.5 cm mean at step 2000, 10.1 cm at the best loss | best val 0.263 at step 1500 |
| [headcam_vision](../headcam_vision/REPORT.md) | 50 | not rolled out | — | — | best val 0.274, worse than the frozen run |
| [vision_xy](../vision_xy/REPORT.md) | 50 layouts | no arm | — | 9.0 cm held-out xy | gate was 5 cm |
| [probes](../probes/REPORT.md) | no new weights | — | — | image readout ties the no-image guess | sentence swap followed the new color 0/5 |

The 10-demo run is the closest reach. Fifty demos of the same frozen features pulled the arm toward a generic reach and finished farther from the ball. The shoulder camera shows the table and still grasped once in twenty. Unfreezing SigLIP on those 50 demos never beat the frozen validation loss, and that checkpoint was never played.

Localization, from [vision_xy](../vision_xy/REPORT.md) and [probes](../probes/REPORT.md):

| Readout | Held-out error |
|---|---|
| Color prior, image ignored | 8.9 cm on the grid frames, 9.6 cm on the mean-pool frames |
| Frozen tokens, shoulder camera, reset | 11.0 cm, against 9.6 cm with no image |
| SigLIP trained, 8×8 grid | 9.0 cm, best at step 1000 |
| SigLIP trained, mean pool | 9.2 cm, best at step 600 |
| Gate | 5 cm |

SmolVLM2-500M, asked to point at the named ball, replied a constant pixel. Overhead: median 170.5 px, 0/20 on the ball. Three images: median 67.6 px, 0/20 on the ball. Replies were `10, 10` or `100, 100` while the ball moved.

The frozen image tokens do not say which ball the sentence named. The action head imitates the reach anyway, because the long motion dominates the loss and the grasp lives in the leftover centimeter.

## Peg in the hole

One peg, one socket, one sentence. That removes the color choice and leaves a 6 mm hole. The expert inserts on 18 of 20 legal layouts, 0.2–1.9 mm from the hole center, after a planning rollout shifts the aim by about 12 mm of actuator sag.

| Run | Demos | Inserted | Grasps | Closest |
|---|---|---|---|---|
| [expert](../peg_socket/REPORT.md#expert) | script | 18/20 | 18/20 | 0.2–1.9 mm on successes |
| [zero-shot](../peg_socket/REPORT.md#zero-shot) | 0 | 0/20 | 0/20 | 4.8 cm best, 10.6 cm median |
| [face probe](../peg_socket/REPORT.md#face-camera) | 10 | 0/5 at steps 500, 1000, and 1500 | 0/5 | 4.2 cm best, 6.7 cm median at the best loss |
| [chunk playback](../peg_socket/REPORT.md#chunk-playback) | same weights | 0/5 on exec 8, 16, and 50 | 0/5 | 4.0 / 6.5 / 6.9 cm best |
| [localize](../peg_socket/REPORT.md#localize) | — | not measured | — | gate is 1 cm median |

The face-camera run is the only peg policy that was trained. `tablecam` looks down from between the shoulders, so the peg, the hole, and the gripper are in one frame. Validation loss fell from 0.409 at step 250 to 0.151 at step 1500 and was still falling. The hand did not get closer. Step 1000 and step 1500 are farther from the peg, in the median, than step 500.

Playing more of each prediction does not close that gap. On the 50-step chunk the median joint-target step inside a prediction is 0.088 rad, the 95th percentile is 0.215 rad, and the largest inside step is 0.74 rad. That inside maximum is larger than the median yank at a replan. The hole still needs about 6 mm.

Step 5 never asked SmolVLM2. The fp32 load ran out of GPU memory while another process held the 6 GB card, so there is no median and no pass. On the balls, that same checkpoint is the 0/20 pointing result, and the peg is the smaller object in the frame. Step 6 stays unstarted. These runs say an action-only fine-tune on `data/datasets/peg_train` would fit the joint targets and still miss the peg.

## What the two tasks share

The action expert copies the demonstration. Seeing the object does not follow from that. Those runs used a GTX 1660: fp32, vision frozen or barely unfrozen, language model frozen, batch 2 once the desktop was using about 1 GB. The machine is now an RTX 3080, 10 GB, which supports bf16 and a larger batch. The part that would have to learn where the peg or the ball is did not get trained on those runs. More demonstrations of the same frozen tower, a different camera, and a longer action chunk are already recorded as misses.

## What would make another training run worth starting

A localizer that beats the centimeter gate on held-out frames, scored before any action head is trained.

- Ball: under 5 cm. Best held-out number is 9.0 cm, tied with ignoring the image.
- Peg: median under 1 cm, and the point on the peg in most frames. Not measured. Do not start step 6 until [localize](../peg_socket/REPORT.md#localize) says pass.

## Sources

Throw: `findings/expert/REPORT.md`, `findings/overfit10/REPORT.md`, `findings/stageA/REPORT.md`, `findings/headcam/REPORT.md`, `findings/headcam_vision/REPORT.md`, `findings/vision_xy/REPORT.md`, `findings/probes/REPORT.md`.

Peg: `peg_socket/REPORT.md`.
