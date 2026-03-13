#!/bin/bash
set -e

cd /home/ubuntu
sudo apt-get update -qq && sudo apt-get install -y -qq git

git clone -b v3-neurogenesis https://github.com/okezue/websleuths-229.git || (cd websleuths-229 && git checkout v3-neurogenesis && git pull)
cd websleuths-229

pip install -e ".[dev,exa,claude]"
pip install openai exa-py bitsandbytes

export OPENAI_API_KEY="$OPENAI_API_KEY"
export EXA_API_KEY="$EXA_API_KEY"
export HF_TOKEN="$HF_TOKEN"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"

nvidia-smi
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

nohup python3 scripts/run_comprehensive_bench.py \
  --model MiniMaxAI/MiniMax-M2.5 \
  --recipe eatrd \
  --lora-r 64 \
  --out /home/ubuntu/wm_v3_bench \
  --exa-key "$EXA_API_KEY" \
  --anthropic-key "$ANTHROPIC_API_KEY" \
  --openai-key "$OPENAI_API_KEY" \
  --gpt-model gpt-5.4 \
  --lr 2e-4 \
  --bs 2 \
  --ml 256 \
  --min-steps 30 \
  --max-steps 100 \
  --eval-n 50 \
  --mmlu-every 2 \
  --example-n 3 \
  --claude-model claude-sonnet-4-5-20250929 \
  --claude-concurrency 8 \
  --distill-n 15 \
  --mu-init 0.5 \
  --neurogenesis \
  --rank-step 8 \
  --max-rank 128 \
  --hf-token "$HF_TOKEN" \
  > /home/ubuntu/bench.log 2>&1 &

echo "Benchmark launched in background. Monitor with: tail -f /home/ubuntu/bench.log"
