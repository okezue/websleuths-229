#!/bin/bash
set -ex

pip install transformers peft datasets accelerate exa_py pydantic scipy tokenizers sentencepiece 2>/dev/null || pip3 install transformers peft datasets accelerate exa_py pydantic scipy tokenizers sentencepiece

cd /home/ubuntu
if [ -d repo ]; then
    cd repo && git pull origin main
else
    git clone https://github.com/okezue/websleuths-229.git repo && cd repo
fi

export EXA_API_KEY="e337f35a-e56c-4ae7-8596-f44959053342"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/ubuntu/repo:$PYTHONPATH

echo "=== starting run_agentic_scale.py ==="
nohup python3 scripts/run_agentic_scale.py > /tmp/wm_run.log 2>&1 &
echo "PID=$!" | tee /tmp/wm_pid
echo "tail -f /tmp/wm_run.log to watch progress"
