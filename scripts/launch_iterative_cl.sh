#!/bin/bash
set -e

AMI="ami-0aad28499825d76c3"
INST="g5.xlarge"
KEY="okezue"
SG="sg-08bbfd4174e4a3665"
REGION="us-east-1"
DISK_GB=150

echo "=== Launching $INST (A10G 24GB) with $AMI ==="

USERDATA=$(cat <<'STARTUP'
#!/bin/bash
set -ex
exec > /tmp/startup.log 2>&1

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pytorch 2>/dev/null || true

pip install -q transformers peft datasets accelerate exa_py pydantic scipy tokenizers sentencepiece

cd /home/ubuntu
git clone https://github.com/okezue/websleuths-229.git repo 2>/dev/null || true
cd repo
git pull origin main 2>/dev/null || true

export EXA_API_KEY="e337f35a-e56c-4ae7-8596-f44959053342"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/ubuntu/repo:$PYTHONPATH

nohup python3 scripts/run_iterative_cl.py \
    --model Qwen/Qwen2.5-3B \
    --recipes eatrd dpmu eab_ssc \
    --steps 50 \
    --bench-n 100 \
    --mmlu-n 50 \
    --lora-r 32 \
    --ckpt-dir /tmp/wm_checkpoints \
    --out /tmp/wm_iterative_cl_results.json \
    > /tmp/wm_iter_cl.log 2>&1 &
echo "PID=$!" > /tmp/wm_pid
echo "started at $(date)" >> /tmp/wm_iter_cl.log
STARTUP
)

ENCODED=$(echo "$USERDATA" | base64)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type "$INST" \
    --key-name "$KEY" \
    --security-group-ids "$SG" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${DISK_GB},\"VolumeType\":\"gp3\"}}]" \
    --user-data "$ENCODED" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-iterative-cl}]" \
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
echo ""
echo "SSH:  ssh -i ~/.ssh/${KEY}.pem ubuntu@${PUBLIC_IP}"
echo "Logs: ssh -i ~/.ssh/${KEY}.pem ubuntu@${PUBLIC_IP} 'tail -f /tmp/wm_iter_cl.log'"
echo "Stop: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""
echo "Results: /tmp/wm_iterative_cl_results.json"
echo "Copy: scp -i ~/.ssh/${KEY}.pem ubuntu@${PUBLIC_IP}:/tmp/wm_iterative_cl_results.json ."
