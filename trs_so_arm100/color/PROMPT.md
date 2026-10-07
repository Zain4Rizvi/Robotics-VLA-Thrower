You are implementing trs_so_arm100/color/PLAN.md. Read that plan, trs_so_arm100/PLAN.md, trs_so_arm100/REPORT.md, trs_so_arm100/view.py, and findings/so_xy_cont/REPORT.md before editing. Follow AGENTS.md.

The best checkpoint is trs_so_arm100/best. Do not delete it, overwrite it, or train into that folder. Vision stays frozen. The language model stays frozen. Do not fine-tune SigLIP. Do not start predicted-xy training. Do not rewrite the robot MJCF physics. Recolor cube geoms in Python at reset. The floor geom stays groundplane.

Run the plan in order. Step 1 is both red/green sentences, state slots fixed to the two bodies. Step 2 paints four trained colors and the sentence names them; do not reorder state into source-then-target. Step 3 only evals the step 2 weights on orange and purple, which must never appear in the training text. Stop when the plan's stop section says to stop. A 0-stack table is a result. Write it.

Each step that trains gets a report, loss.png, success.png, and the videos the plan names, before the next step starts. Do not claim a stack without that step's eval_summary.json. Counts come from the rollout, not from the loss.

GTX 1660. fp32. One GPU job. Batch 2 if the desktop is already on the card, otherwise 8. Nothing on C:. First line of every new shell:

$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"

Launch each long train with Start-Process, hidden, logs inside that step's folder. Keep going until the plan's last report is written or a stop condition in the plan is hit.
