conda activate torch-cuda
$env:PYTHONPATH="."; python .\tools\start_prompt_server.py; $env:PYTHONPATH=""
Read-Host -Prompt "Press Enter to exit"
