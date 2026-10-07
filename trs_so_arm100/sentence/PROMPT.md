You are implementing trs_so_arm100/sentence/PLAN.md. Read that plan, trs_so_arm100/PLAN.md, trs_so_arm100/REPORT.md, trs_so_arm100/view.py, trs_so_arm100/color/swap/REPORT.md, and findings/smolvla_architecture.md before editing. Follow AGENTS.md.

The best checkpoint is trs_so_arm100/best. Do not delete it, overwrite it, or train into that folder. Do not init from trs_so_arm100/color/swap/checkpoints. Do not fine-tune SigLIP. Do not start predicted-xy training. Do not collect a new dataset. The swap seeds are already in trs_so_arm100/color/swap/datasets. Do not reorder state into source-then-target. Do not rewrite the robot MJCF physics. The floor geom stays groundplane.

Run the probe first. Write trs_so_arm100/sentence/probe/REPORT.md with accuracy.png, distance.png, notes.png, and the frame stills before any training. The probe gate picks one train. Step 2 writes the frozen classifier logits into state dims 10:12 and trains the action expert only. Step 3 unfreezes the text layers, keeps SigLIP and the connector frozen, and leaves state dims 10:32 at zero. Run one of those. Leave the other unstarted. If the probe lands between the two gates, stop after the probe report.

A 0-stack table is a result. Write it. Do not claim a stack without that step's eval_summary.json. Counts come from the rollout, not from the loss. Accuracies come from probe.json.

RTX 3080, 10 GB. bf16 is supported. The load path still casts the policy to fp32. Leave that cast. One GPU job. Pass a batch that fits free VRAM. Expert-only training starts at batch 8. The text-layer train starts at batch 4. The old batch-2 rule was the 6 GB card. Nothing on C:. First line of every new shell:

$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"

Launch each long train with Start-Process, hidden, logs inside that step's folder. Keep going until the plan's last report is written or a stop condition in the plan is hit.
