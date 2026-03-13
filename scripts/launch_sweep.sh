#!/bin/bash
set -euo pipefail

INSTANCE_TYPE="${1:-g5.xlarge}"
AMI="ami-0aad28499825d76c3"
KEY="wm-gpu-key"
SG="sg-08bbfd4174e4a3665"
REGION="us-east-1"
BRANCH="v3-neurogenesis"
REPO="https://github.com/okezue/websleuths-229.git"

echo "=== Launching $INSTANCE_TYPE for authority sweep ==="

ID=$(aws ec2 run-instances \
  --region $REGION \
  --image-id $AMI \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY \
  --security-group-ids $SG \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-authority-sweep}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance: $ID"
echo "Waiting for running..."
aws ec2 wait instance-running --instance-ids $ID --region $REGION

IP=$(aws ec2 describe-instances --instance-ids $ID --region $REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "IP: $IP"

echo "Waiting for SSH..."
for i in $(seq 1 30); do
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/$KEY.pem ubuntu@$IP "echo ready" 2>/dev/null && break
  sleep 10
done

ssh -i ~/.ssh/$KEY.pem ubuntu@$IP <<'REMOTE'
source /opt/pytorch/bin/activate
pip install -q peft datasets exa_py anthropic openai crochet scrapy pydispatcher networkx scikit-learn circuit_tracer 2>/dev/null
cd /home/ubuntu
git clone REPO_PLACEHOLDER -b BRANCH_PLACEHOLDER websleuths-229 || (cd websleuths-229 && git pull)
cd websleuths-229

git clone https://github.com/wzzll123/MultiKernelBench.git /home/ubuntu/MultiKernelBench 2>/dev/null || true
export MULTIKERNELBENCH_PATH=/home/ubuntu/MultiKernelBench

nohup python scripts/run_authority_sweep.py \
  --model meta-llama/Llama-3.2-1B \
  --out /home/ubuntu/authority_sweep \
  --eval-n 50 \
  --max-steps 200 \
  --lr 3e-4 \
  --lora-r 32 \
  > /home/ubuntu/sweep.log 2>&1 &

echo "PID: $!"
echo "Logs: tail -f /home/ubuntu/sweep.log"
REMOTE

sed -i '' "s|REPO_PLACEHOLDER|$REPO|g" /dev/null 2>/dev/null || true
echo ""
echo "=== LAUNCHED ==="
echo "Instance: $ID"
echo "IP: $IP"
echo "SSH: ssh -i ~/.ssh/$KEY.pem ubuntu@$IP"
echo "Logs: ssh -i ~/.ssh/$KEY.pem ubuntu@$IP 'tail -f /home/ubuntu/sweep.log'"
