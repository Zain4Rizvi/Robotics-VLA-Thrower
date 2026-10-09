# Stacking playground

One local page for the red/green stack. The policy runs the arm. The page sets the instruction, draws a new layout, and shows the MuJoCo cameras.

Z: is not mounted here, so caches stay on D:.

From the repo root, in a terminal you keep open:

```powershell
$env:HF_HOME="D:\hf_cache"; $env:UV_CACHE_DIR="D:\uv_cache"; $env:TEMP="D:\tmp"; $env:TMP="D:\tmp"; $env:TORCH_HOME="D:\torch"; $env:XDG_CACHE_HOME="D:\xdg"; $env:MUJOCO_GL="glfw"
uv run python trs_so_arm100/playground/serve.py
```

That process is the arm. It loads the policy, runs the sim, and serves the page. Open http://127.0.0.1:8765

Stop it with Ctrl+C in that same terminal. The page and the arm both exit. Closing the browser does not stop the arm.

If the terminal is already gone and the arm is still running:

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

Use one browser tab. A second tab takes over the connection.

After a UI change, rebuild `trs_so_arm100/playground/web` with `npm run build`, then start the server again.
