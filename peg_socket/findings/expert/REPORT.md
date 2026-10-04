# expert

Status: done

## Result

The open-loop expert inserted the peg on 18 of 20 legal layouts. That meets the 18/20 bar, so step 3 may start. Both failures are `never_grasped`: the approach knocked the peg over before the fingers closed. No trial dropped the peg after a grasp, missed the hole after a release, or timed out.

## Process

From the repo root, after the cache env vars:

```
uv run python peg_socket/eval_peg_expert.py --n 20 --max-seeds 400
```

`peg_socket/peg_expert.py` plans from the true peg and socket poses with `openarm_vla/expert/ik.py`, then replays joint targets. Phases are hover, open, descend, close, lift, carry, lower through the hole, open, retreat. The lower stops while the finger meshes are still above the octagon; the peg is already in the hole and drops the last centimetre. A planning rollout measures where the hand actually sits (gravity sag on the position actuators is about 12 mm) and shifts the aim before the replay.

Seeds are `0, 1, 2, ...`. Each seed draws one peg xy and one socket xy with `numpy` Generator uniform in x `[0.18, 0.36]`, y `[-0.45, -0.18]`. The first 20 legal draws are the attempts. Seeds scanned: 0 through 89.

A draw is legal when all of these hold:

- both objects are upright on the tabletop (table center `(0.47, 0)`, half-extents `(0.41, 0.55)`, top at z `0.40`)
- at least 1 cm clear of the pedestal and of either gripper at the home pose
- peg center at least 0.12 m from the socket center, so the open fingers clear the octagon
- the peg is not already in the hole
- the same downward yaw reaches the hover, grasp, lift, and carry poses within 3 mm

70 draws were rejected: 48 too close, 14 inside the home gripper or pedestal, 2 knocked during the home settle, 6 unreachable. Inserted means the peg center stays within 7 mm of the hole center, between 40 mm and 51 mm above the socket origin, and upright, for 0.5 s after the fingers open.

## Numbers

| | |
|---|---|
| Inserted | 18 |
| Attempts | 20 |
| Seeds | 0, 3, 5, 11, 13, 15, 18, 22, 23, 26, 36, 47, 52, 58, 62, 64, 66, 75, 81, 89 |

Quote from `summary.json`:

```json
"inserted": 18,
"attempts": 20,
"insertions": "18/20"
```

Successes finished 0.2 mm to 1.9 mm from the hole center. The socket did not slide.

| Seed | Cause | Reason |
|---|---|---|
| 62 | never_grasped | fingers never both touched the peg (closest 84 mm, socket drift 0 mm) |
| 89 | never_grasped | fingers never both touched the peg (closest 149 mm, socket drift 0 mm) |

## Failures other agents should know

Seed 62 started with the peg at `(0.290, -0.248)` and seed 89 at `(0.284, -0.242)`. Both ended on their side (center z `0.414`, which is the table plus the peg radius). That spot is under the throw-ready transit the approach still uses, whose grasp point is about `(0.24, -0.23, 0.53)`. The arm knocks the peg over on the way to the hover.

Replacing that transit with one high waypoint, solved on the same yaw as the grasp, fixed seed 62 and made seed 0 unreachable. Do not retry that. Excluding a disk around `(0.24, -0.23)` from the draw is the change that matches these two failures. Leave the sag-correction loop in place: kinematic IK alone sits about 12 mm off in x, and the peg then hits the octagon.

The scene copy was not edited.

## Sources

- `peg_socket/findings/expert/summary.json`
- `peg_socket/findings/expert/eval.log`
- `peg_socket/peg_expert.py`
- `peg_socket/eval_peg_expert.py`
