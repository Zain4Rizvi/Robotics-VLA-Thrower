# VLA training findings

This folder is the record of the SmolVLA fine-tunes on the OpenArm throw task, plus the scripted expert those runs imitated. The loss plots, the closed-loop counts, and the videos in each subfolder are enough to see what was tried and what the robot did.

The layer widths of the network those runs trained are in [smolvla_architecture](smolvla_architecture.md).

The task is a sentence such as "throw the red ball into the blue bucket." The right arm has to pick that ball off the side table and throw it into that floor bin. Five balls and five bins, colors shuffled independently. The policy sees two cameras, its own joints, and the sentence. It never sees ball or bin positions. The scripted expert does, and the training data is that expert's successful throws.

## How a number was measured

Closed-loop eval plays the policy in the simulator. Physics pauses while the network predicts the next joint targets.

- **Success.** The target ball's center stays inside the target bin for half a second. The bin check is applied even when the fingers never registered a grasp, so a "success" video can be a ball that ended in the bin without a clean pick. The reports say when that happened.
- **Grasp.** The finger-contact flag was true on some step.
- **Closest.** Average, over episodes, of the nearest the gripper control point came to the target ball. A pinch is about a centimeter. 6–16 cm is a miss. Lower is a closer reach.
- **Flow-matching loss.** How closely the action head matches the expert's joint targets. The long reach dominates it. A low loss means the arm motion looks like the demos. It does not mean the ball was picked up.

Checkpoints inside one run share scenes, so they compare with each other. Overfit10 and Stage A share the same 20 room-camera scenes. The head-camera runs use a different camera and different scenes. Their centimeter numbers are a different test.

The policy samples noise when it predicts, so the same weights rolled out twice do not match. On 20 episodes, one extra grasp moves the rate by 5 points.

Training stopped on its own when validation loss failed to improve by more than 2% for four checks in a row. That rule is the dotted line on the later loss plots.

## What happened

| Folder | What changed | Robot result |
|---|---|---|
| [expert](expert/REPORT.md) | scripted throw, the data the network imitated | 87/100 land in the bin |
| [overfit10](overfit10/REPORT.md) | 10 demos, vision frozen, 2000 steps, room camera | 0/20 in the bin, 3 grasps, 5.8 cm |
| [stageA](stageA/REPORT.md) | 50 demos, same frozen recipe | same 20 scenes: 1/20 in the bin, 1 grasp, 8.4 cm |
| [headcam](headcam/REPORT.md) | shoulder camera, 50 new demos, vision still frozen | new 20 scenes: 0/20 in the bin, 1 grasp, 10.1 cm at the best loss |
| [headcam_vision](headcam_vision/REPORT.md) | SigLIP unfrozen, language model still frozen | no robot eval; validation loss stayed worse than the frozen run |
| [probes](probes/REPORT.md) | no new weights | frozen vision cannot find the ball; the sentence does not steer the arm |

The 10-demo run is the closest reach that was measured. More demos, a new camera, and unfreezing the vision tower did not produce a policy that throws.

The pattern across the probes: the action head imitates a reach, and the frozen image and language features do not tell it which ball the sentence named. The loss can fall two orders of magnitude while the hand stays several centimeters off the ball.

## Camera warning

Overfit10 and Stage A were trained and filmed with a room camera. The simulator was later changed so the front image is a shoulder camera named `headcam`. The videos in those two folders are the original recordings. Evaluating an old checkpoint in the current simulator would show it a view it never trained on.

## What this folder leaves out

`checkpoints/bench` is a 100-step plumbing check on 3 demos, and `artifacts/smolvla_smoke` is a one-episode smoke eval. Neither is a fine-tune result. A 500-demo set was never collected. The vision-unfreeze checkpoint was never played on the robot, so that folder has a loss plot and no videos.

Every seed's video, if you need the full set, is under `artifacts/`. The clips here are the ones that show the result.
