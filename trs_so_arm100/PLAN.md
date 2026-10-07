# Keep this checkpoint

`best/` is the best model so far. It stacks 5 of 10 held-out layouts when it is given the true cube positions. Read `REPORT.md` and [findings/so_xy_cont](../findings/so_xy_cont/REPORT.md) before changing the policy or the scene. Do not delete or overwrite `best/`. Later runs get their own output directory.

What is already settled, in `findings/`:

- A scripted expert can stack. Cameras and joints alone, with vision frozen, stacked 0 of 10 ([so_stack](../findings/so_stack/REPORT.md)).
- Frozen SigLIP places each cube to about 1.6 cm. Training SigLIP on these 50 demos made that worse ([so_see](../findings/so_see/REPORT.md)).
- Giving the action expert true red xy and green xy in state dimensions 6:10, then training it long enough, reached 5 of 10 ([so_xy_cont](../findings/so_xy_cont/REPORT.md)). Vision and language stayed frozen.

The next sitting is `color/PLAN.md`: both red/green sentences, then other color words, still with true cube positions and with vision frozen. A vision readout that replaces those positions stays later. Predicted-xy training waits on that readout. Another SigLIP fine-tune on these same 50 demos is the attempt that already got worse.
