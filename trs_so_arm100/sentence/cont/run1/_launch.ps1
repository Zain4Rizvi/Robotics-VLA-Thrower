$env:HF_HOME="C:\Users\Zain\.cache\huggingface"
$env:HUGGINGFACE_HUB_CACHE="C:\Users\Zain\.cache\huggingface\hub"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:UV_CACHE_DIR="D:\uv_cache"
$env:TEMP="D:\tmp"
$env:TMP="D:\tmp"
$env:TORCH_HOME="D:\torch"
$env:XDG_CACHE_HOME="D:\xdg"
$env:MUJOCO_GL="glfw"

$p = Start-Process -FilePath 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\.venv\Scripts\python.exe' -ArgumentList @('-u', 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\sentence\cont\eval.py', '--out', 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\sentence\cont\run1', '--init', 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\sentence\use_note\checkpoints\checkpoints\002000\pretrained_model', '--ckpt-root', 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\sentence\cont\run1\checkpoints', '--seeds-file', 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\color\swap\datasets\train\stack_seeds.json', '--seeds-file', 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\color\swap\datasets\val\stack_seeds.json', '--require', '0,2000,4000,6000,8000') -WorkingDirectory 'C:\Users\Zain\Documents\Robotics-VLA-Thrower' -RedirectStandardOutput 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\sentence\cont\run1\eval_stdout.txt' -RedirectStandardError 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\sentence\cont\run1\eval_stderr.txt' -WindowStyle Hidden -PassThru
Set-Content -Path 'C:\Users\Zain\Documents\Robotics-VLA-Thrower\trs_so_arm100\sentence\cont\run1\eval_pid.txt' -Value $p.Id
$null = $p.WaitForExit()
if ($null -eq $p.ExitCode) { exit 1 }
exit $p.ExitCode
