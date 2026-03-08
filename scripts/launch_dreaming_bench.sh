#!/bin/bash
set -e

AMI="ami-0aad28499825d76c3"
INST="${INSTANCE_TYPE:-g5.2xlarge}"
KEY="${KEY_NAME:-okezue}"
SG="${SECURITY_GROUP:-sg-08bbfd4174e4a3665}"
REGION="${AWS_REGION:-us-east-1}"
DISK_GB="${DISK_SIZE:-200}"
RECIPE="${RECIPE:-dpmu}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
EVAL_N="${EVAL_N:-0}"
MMLU_EVERY="${MMLU_EVERY:-2}"
EXA="${EXA_API_KEY:-e337f35a-e56c-4ae7-8596-f44959053342}"
HF="${HF_TOKEN:-}"

echo "=== Dreaming++ Benchmark ==="
echo "Instance: $INST  AMI: $AMI  Disk: ${DISK_GB}GB"
echo "Recipe: $RECIPE  Model: $MODEL  eval_n=$EVAL_N"
echo ""

USERDATA=$(cat <<'STARTUP'
#!/bin/bash
set -ex
exec > /tmp/startup.log 2>&1

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pytorch 2>/dev/null || true

pip install -q transformers peft datasets accelerate exa_py pydantic scipy tokenizers sentencepiece matplotlib

cd /home/ubuntu
git clone https://github.com/okezue/websleuths-229.git repo 2>/dev/null || true
cd repo
git pull origin main 2>/dev/null || true

export EXA_API_KEY="__EXA__"
export HF_TOKEN="__HF__"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/ubuntu/repo:$PYTHONPATH

NUM_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs detected: $NUM_GPU"

nohup python3 scripts/run_dreaming_bench.py \
    --model "__MODEL__" \
    --recipe "__RECIPE__" \
    --lora-r 32 \
    --out /tmp/wm_dreaming_bench \
    --lr 2e-4 \
    --bs __BS__ \
    --ml 256 \
    --min-steps 30 \
    --max-steps 150 \
    --eval-n __EVAL_N__ \
    --mmlu-every __MMLU_EVERY__ \
    --example-n 5 \
    > /tmp/wm_dreaming_bench/bench.log 2>&1 &

echo "PID=$!" > /tmp/wm_pid
echo "started at $(date)" >> /tmp/wm_dreaming_bench/bench.log
STARTUP
)

BS=2
case "$INST" in
    g5.xlarge)  BS=2;;
    g5.2xlarge) BS=4;;
    g5.4xlarge) BS=4;;
    g5.12xlarge) BS=8;;
    g5.48xlarge) BS=8;;
    p4d*|p4de*) BS=16;;
    p5*) BS=16;;
    *) BS=2;;
esac

USERDATA=$(echo "$USERDATA" | sed "s|__EXA__|$EXA|g")
USERDATA=$(echo "$USERDATA" | sed "s|__HF__|$HF|g")
USERDATA=$(echo "$USERDATA" | sed "s|__MODEL__|$MODEL|g")
USERDATA=$(echo "$USERDATA" | sed "s|__RECIPE__|$RECIPE|g")
USERDATA=$(echo "$USERDATA" | sed "s|__BS__|$BS|g")
USERDATA=$(echo "$USERDATA" | sed "s|__EVAL_N__|$EVAL_N|g")
USERDATA=$(echo "$USERDATA" | sed "s|__MMLU_EVERY__|$MMLU_EVERY|g")

ENCODED=$(echo "$USERDATA" | base64)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INST" \
    --key-name "$KEY" \
    --security-group-ids "$SG" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${DISK_GB},\"VolumeType\":\"gp3\"}}]" \
    --user-data "$ENCODED" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-dreaming-bench-${RECIPE}}]" \
    --query 'Instances[0].InstanceId' \
    --output text \
    --region "$REGION")

echo "Instance: $INSTANCE_ID"
echo "Waiting for running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region "$REGION")

echo ""
echo "=== INSTANCE READY ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo "Type:        $INST"
echo "Recipe:      $RECIPE"
echo "Model:       $MODEL"
echo "Batch size:  $BS"
echo "Eval:        $([ "$EVAL_N" = "0" ] && echo 'FULL' || echo "sample $EVAL_N")"
echo ""
echo "SSH:    ssh -i ~/.ssh/${KEY}.pem ubuntu@${PUBLIC_IP}"
echo "Logs:   ssh -i ~/.ssh/${KEY}.pem ubuntu@${PUBLIC_IP} 'tail -f /tmp/wm_dreaming_bench/bench.log'"
echo "Stop:   aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "Copy results:"
echo "  scp -i ~/.ssh/${KEY}.pem -r ubuntu@${PUBLIC_IP}:/tmp/wm_dreaming_bench/ ./dreaming_bench_results/"
echo ""
echo "Multi-GPU instances available:"
echo "  INSTANCE_TYPE=g5.12xlarge bash $0   # 4x A10G (96GB)"
echo "  INSTANCE_TYPE=p4d.24xlarge bash $0  # 8x A100 (320GB)"
