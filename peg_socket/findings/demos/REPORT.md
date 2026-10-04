# demonstrations

Status: done

## Result

`data/datasets/peg_train` has 50 success episodes and `data/datasets/peg_val` has 20. The saved seed lists are disjoint: train seeds are in 1003–1271 and val seeds are in 5002–5108. Step 6 may train on these sets. Step 6 still waits on step 5. This step is done because both counts are 50 and 20.

## Process

CPU collectors only. SmolVLA was not loaded. The expert in `peg_socket/peg_expert.py` was not changed: the throw-ready approach transit and the sag-correction loop are the ones that scored 18/20.

Each seed is one `numpy` Generator draw, the same function as step 1 (`eval_peg_expert.draw` / `prepare`). The stream for train starts at 1000 and stops before 5000. The stream for val starts at 5000 and stops before 9000. Train used 1000–1271. Val used 5000–5108.

A draw is kept only when the step 1 rules hold and, in addition, the peg cylinder and the socket wall boxes project inside `tablecam` with a 16 px margin on the 256 image (pixels 16 through 239 inclusive). That camera reject is counted as `tablecam_edge`, separate from the step 1 rejects. Obs is stored before the action. Success is the peg center inside the socket for 0.5 s after the fingers open. Failed attempts are not saved. `ball_color` is `peg` and `bin_color` is `socket` so `scripts/dataset_stats.py` can read `openarm_seeds.json`. The task string is `place the peg in the hole`. Videos use pyav.

```
.venv\Scripts\python.exe peg_socket/collect_peg_demos.py --n-success 50 --seed-start 1000 --seed-end 5000 --out data/datasets/peg_train --tag train
.venv\Scripts\python.exe peg_socket/collect_peg_demos.py --n-success 20 --seed-start 5000 --seed-end 9000 --out data/datasets/peg_val --tag val
uv run python scripts/dataset_stats.py --dataset data/datasets/peg_train
uv run python scripts/dataset_stats.py --dataset data/datasets/peg_val
```

Nothing was frozen. This is the scripted expert.

The legal draw, which a later agent must reuse:

- uniform x `[0.18, 0.36]`, y `[-0.45, -0.18]`
- both objects upright on the tabletop, at least 1 cm clear of the pedestal and of either gripper at the home pose
- peg center at least 0.12 m from the socket center
- peg not already in the hole
- the same downward yaw reaches hover, grasp, lift, and carry within 3 mm
- both objects project inside `tablecam` with a 16 px margin on the 256 image

## Numbers

| | Train | Val |
|---|---|---|
| Successes saved | 50 | 20 |
| Seeds scanned | 1000–1271 (272) | 5000–5108 (109) |
| too_close | 150 | 59 |
| pedestal_or_gripper | 51 | 16 |
| settled_pedestal_or_gripper | 1 | 2 |
| unreachable | 17 | 12 |
| tablecam_edge | 0 | 0 |
| never_grasped (not saved) | 1 | 0 |
| missed (not saved) | 2 | 0 |
| dropped | 0 | 0 |
| timeout | 0 | 0 |

`tablecam_edge` is zero because this xy box already sits inside the margin. The filter does reject: a peg at y `-0.55` projects to v `243` and is `tablecam_edge`. The peg-center projection on seed 1003 matched the segmented peg within 0.65 px.

`dataset_stats` on `data/datasets/peg_train`:

```
{
  "n_episodes": 50,
  "n_frames": 40649,
  "length_min": 690,
  "length_max": 863,
  "length_mean": 812.98,
  "action_min": [
    -1.396299958229065,
    2.2211752366274595e-05,
    -1.0770480632781982,
    0.40725451707839966,
    -1.5707999467849731,
    -0.7853999733924866,
    -1.5707999467849731,
    -0.7853999733924866
  ],
  "action_max": [
    -0.014991266652941704,
    3.2887260913848877,
    0.29097700119018555,
    2.4000000953674316,
    1.237589955329895,
    0.7799999713897705,
    0.5587760210037231,
    0.0
  ],
  "action_mean": [
    -0.8364132724904161,
    2.5575512318204217,
    -0.42299723909136944,
    1.3927034805300738,
    -1.2897506942666674,
    0.11905513046176902,
    -0.7521793379286899,
    -0.19997769447193423
  ],
  "action_std": [
    0.2674622518035872,
    0.5870849302414538,
    0.28831824042965615,
    0.4482790403722976,
    0.5855606940617019,
    0.30348180771866845,
    0.4072690262624066,
    0.3421565410286586
  ],
  "color_pairs": {
    "peg->socket": 50
  },
  "n_unique_pairs": 1
}
WARNING: color pairs look collapsed; check independent randomization.
```

`dataset_stats` on `data/datasets/peg_val`:

```
{
  "n_episodes": 20,
  "n_frames": 15878,
  "length_min": 718,
  "length_max": 849,
  "length_mean": 793.9,
  "action_min": [
    -1.351767659187317,
    2.2211752366274595e-05,
    -1.0722568035125732,
    0.5914815068244934,
    -1.5707999467849731,
    -0.7402684092521667,
    -1.5087271928787231,
    -0.7853999733924866
  ],
  "action_max": [
    -0.014991266652941704,
    3.110474109649658,
    0.29097700119018555,
    2.4000000953674316,
    1.237589955329895,
    0.7799999713897705,
    0.5587760210037231,
    0.0
  ],
  "action_mean": [
    -0.8585142914737438,
    2.5624043208133185,
    -0.4100453504361799,
    1.3335827628676367,
    -1.2605207995047631,
    0.07386953146350285,
    -0.788824114571712,
    -0.2047838089875722
  ],
  "action_std": [
    0.2684687356119619,
    0.5842249169261278,
    0.2697200133843381,
    0.41307572838833,
    0.587068978074355,
    0.2993837713865172,
    0.40837902676878685,
    0.34481949189250405
  ],
  "color_pairs": {
    "peg->socket": 20
  },
  "n_unique_pairs": 1
}
WARNING: color pairs look collapsed; check independent randomization.
```

The warning is the fixed instruction. There is one task, so one pair. It is not a failed randomization of colors.

## Failures other agents should know

Do not replace the approach transit with one high waypoint. That fixed seed 62 and made seed 0 unreachable. Do not exclude a disk around `(0.24, -0.23)`. Those knockovers are failed attempts and were not saved (train: one `never_grasped`, two `missed`). Leave the sag-correction loop in place.

Starting both collectors at once failed. The val video workers died with `WinError 1455` (paging file too small) while importing torch, and that partial `peg_val` was deleted. Val was collected after train had exited. Do not run two collectors together on this machine, and do not run a collector during a GPU job.

Reuse the bounds in Process, including the `tablecam` 16 px margin. `tablecam` is not `headcam`. Do not point later training at throw checkpoints or at `data/datasets/train_head`.

## Sources

- `data/datasets/peg_train`
- `data/datasets/peg_train/openarm_seeds.json`
- `data/datasets/peg_val`
- `data/datasets/peg_val/openarm_seeds.json`
- `peg_socket/findings/demos/train_stats.txt`
- `peg_socket/findings/demos/val_stats.txt`
- `peg_socket/findings/demos/train_rejects.json`
- `peg_socket/findings/demos/val_rejects.json`
- `peg_socket/findings/demos/train_stdout.txt`
- `peg_socket/findings/demos/val_stdout.txt`
