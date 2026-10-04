# vision_xy

SigLIP was trained to predict the named ball's position on the table. The action expert was not trained. On the 20 held-out reset frames the best miss was 9.2 cm, which is the same miss as ignoring the picture. The gate to continue was 5 cm. No checkpoint was saved, and the arm was not rolled out.

## Recipe

Started from `lerobot/smolvla_base`. Same 50 training episodes and 20 validation episodes as the head-camera runs (`data/datasets/train_head`, `data/datasets/val_head`). The language model, the connector, and the action expert stayed frozen. Only the SigLIP tower was trained, plus a linear readout that is thrown away after the measurement.

The readout averages the 64 front-camera tokens into one vector, appends a one-hot of the named ball's color, and predicts xy. Learning rate `1.0e-5` on SigLIP and `1.0e-3` on the readout. Batch 4. The front camera is the live overhead `headcam`. The wrist camera was not used.

The datasets do not store ball positions. Each seed was replayed with the stored actions. Joint state matched the recording exactly, and the wrist frames stayed within about 4 gray levels, so the positions belong to those episodes. The stored front video does not match the live overhead camera (about 48 gray levels on the first training frame). Training used fresh renders from the live camera, not that stored video.

Training stopped at step 1400, when the held-out error had failed to improve by 0.3 cm for eight checks. The best check was step 600.

## The error

![Held-out reset error](val_cm.png)

Plotted from `artifacts/vision_xy/train_log.csv`. The dashed line is the 5 cm gate. The dotted line is 9.6 cm, the frozen-token probe's score for guessing from the color and ignoring the image.

| Step | Held-out reset error |
|---|---|
| 100 | 9.7 cm |
| 200 | 9.9 cm |
| 300 | 10.7 cm |
| 400 | 12.9 cm |
| 500 | 13.5 cm |
| 600 | 9.2 cm, the best |
| 700 | 9.3 cm |
| 800 | 12.8 cm |
| 900 | 11.2 cm |
| 1000 | 10.0 cm |
| 1100 | 15.5 cm |
| 1200 | 16.4 cm |
| 1300 | 10.6 cm |
| 1400 | 11.7 cm |

Training error on the 50 episodes fell to about a millimeter. The 20 new layouts did not.

## The robot

No closed-loop eval was run. There is no `eval_summary.json`. The action expert is still the SO-100 head from `smolvla_base`.

## Why this happened

Fifty layouts were enough to memorize. They were not enough to report the ball's position on a layout the tower had not seen. The best held-out number, 9.2 cm, sits on the color-only guess from [probes](../probes/REPORT.md).

This is a different failure from [headcam_vision](../headcam_vision/REPORT.md). That run trained SigLIP with the arm's joint loss and judged it by that loss. This run asked for the ball's xy directly and judged it by centimeters on new reset frames.

The average is a crude question. The ball's position can sit in one of the 64 tokens, and the average throws that away. This run shows the average does not generalize. It does not show that a reader which keeps the grid would also miss.

Sources: `artifacts/vision_xy/probe.json`, `artifacts/vision_xy/train_log.csv`, `scripts/train_vision_xy.py`, `scripts/label_ball_xy.py`.
