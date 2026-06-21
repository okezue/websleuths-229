#!/usr/bin/env bash
# trace_aws_launch.sh — launch one EC2 g5 instance, ship the TRACE branch, run the 64-episode
# micro-web stream, and SSH-tunnel the Aim dashboard back to localhost:43800.
#
# Requires on the laptop:
#   aws CLI configured with the okezue account
#   private key at ~/.ssh/wm-gpu-key.pem (chmod 400)
#   env vars: HF_TOKEN (only if using gated model), OPENAI_API_KEY (optional), ANTHROPIC_API_KEY (optional)
#
# Usage:
#   ./scripts/trace_aws_launch.sh                  # launches g5.xlarge, runs stream_350m
#   INSTANCE_TYPE=g5.2xlarge ./scripts/trace_aws_launch.sh
#   CONFIG=configs/stream_full.yaml ./scripts/trace_aws_launch.sh
#   ACTION=stream|baseline-matrix|smoke ./scripts/trace_aws_launch.sh
#   IID=i-0abc... ./scripts/trace_aws_launch.sh attach   # re-attach tunnel + log tail to existing box
#   IID=i-0abc... ./scripts/trace_aws_launch.sh fetch    # rsync runs/ + .aim back to laptop
#   IID=i-0abc... ./scripts/trace_aws_launch.sh terminate
set -euo pipefail

REGION="${REGION:-us-east-1}"
AMI_ID="${AMI_ID:-ami-0aad28499825d76c3}"
KEY_NAME="${KEY_NAME:-wm-gpu-key}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/wm-gpu-key.pem}"
SG_ID="${SG_ID:-sg-08bbfd4174e4a3665}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
DISK_GB="${DISK_GB:-200}"
CONFIG="${CONFIG:-configs/stream_350m.yaml}"
ACTION="${1:-launch}"
N_EPISODES="${N_EPISODES:-64}"
RUN_TAG="${RUN_TAG:-trace-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_HOME="/home/ubuntu"
REMOTE_TRACE="$REMOTE_HOME/trace"
REMOTE_AIM="$REMOTE_HOME/aim_repo"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_TRACE="$LOCAL_REPO/websleuth_models_trace"
LOCAL_RESULTS="$LOCAL_REPO/trace_results/$RUN_TAG"
LOCAL_AIM="$LOCAL_REPO/aim_repos/trace"
SSH_OPTS="-i $KEY_PATH -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o UserKnownHostsFile=$HOME/.ssh/known_hosts_trace"

die(){ echo "[FATAL] $*" >&2; exit 1; }
log(){ echo "[$(date +%H:%M:%S)] $*"; }
need(){ command -v "$1" >/dev/null || die "missing tool: $1"; }

[[ -f "$KEY_PATH" ]] || die "SSH key not found at $KEY_PATH"
[[ -d "$LOCAL_TRACE/src/wm" ]] || die "TRACE source missing at $LOCAL_TRACE"
need aws; need ssh; need rsync

resolve_iid_to_ip(){
  local iid="$1"
  aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
    --query 'Reservations[].Instances[].PublicIpAddress' --output text
}

resolve_iid_to_state(){
  local iid="$1"
  aws ec2 describe-instances --region "$REGION" --instance-ids "$iid" \
    --query 'Reservations[].Instances[].State.Name' --output text
}

launch_instance(){
  log "launching $INSTANCE_TYPE from $AMI_ID in $REGION ($RUN_TAG)"
  local userdata
  userdata=$(cat <<'EOF'
#!/bin/bash
exec > /tmp/startup.log 2>&1
set -e
source /opt/pytorch/bin/activate || true
mkdir -p /home/ubuntu/trace /home/ubuntu/aim_repo /home/ubuntu/hf_cache /home/ubuntu/.cache/huggingface
chown -R ubuntu:ubuntu /home/ubuntu/trace /home/ubuntu/aim_repo /home/ubuntu/hf_cache /home/ubuntu/.cache
echo READY > /tmp/boot_ready
EOF
)
  local iid
  iid=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wm-$RUN_TAG},{Key=project,Value=websleuths-trace}]" \
    --user-data "$userdata" \
    --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=2" \
    --query 'Instances[0].InstanceId' --output text)
  echo "$iid" > "$LOCAL_REPO/.trace_aws_iid"
  log "instance id: $iid (saved to .trace_aws_iid)"
  log "waiting for running state"
  aws ec2 wait instance-running --region "$REGION" --instance-ids "$iid"
  local ip; ip=$(resolve_iid_to_ip "$iid")
  log "instance running at $ip — waiting for SSH"
  for _ in $(seq 1 60); do
    ssh $SSH_OPTS -o ConnectTimeout=5 ubuntu@"$ip" "test -f /tmp/boot_ready" 2>/dev/null && break
    sleep 5
  done
  ssh $SSH_OPTS ubuntu@"$ip" "test -f /tmp/boot_ready" || die "boot_ready never appeared"
  log "boot ready"
  echo "$iid:$ip"
}

push_code(){
  local ip="$1"
  log "rsyncing trace branch to ubuntu@$ip:$REMOTE_TRACE"
  rsync -az --delete -e "ssh $SSH_OPTS" \
    --exclude '.git' --exclude 'runs/' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.pytest_cache' --exclude '.aim/' \
    "$LOCAL_TRACE/" "ubuntu@$ip:$REMOTE_TRACE/"
}

install_and_run(){
  local ip="$1"
  local action="$2"
  log "installing extras and starting $action on $ip"
  ssh $SSH_OPTS ubuntu@"$ip" \
    HF_TOKEN="${HF_TOKEN:-}" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    RUN_TAG="$RUN_TAG" \
    N_EPISODES="$N_EPISODES" \
    CONFIG="$CONFIG" \
    ACTION_NAME="$action" \
    bash -s <<'REMOTE'
set -euo pipefail
source /opt/pytorch/bin/activate
cd /home/ubuntu/trace
export HF_HOME=/home/ubuntu/hf_cache
export HF_TOKEN="${HF_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
pip install -q -e '.[models,aim,media]' || pip install -q -e '.[models,aim]'
mkdir -p /home/ubuntu/aim_repo data
if ! aim --repo /home/ubuntu/aim_repo status >/dev/null 2>&1; then
  aim init --repo /home/ubuntu/aim_repo >/dev/null 2>&1 || true
fi
# kill any prior aim up + experiment for re-runs
tmux kill-session -t aim 2>/dev/null || true
tmux kill-session -t trace 2>/dev/null || true
tmux new -d -s aim "source /opt/pytorch/bin/activate && aim up --repo /home/ubuntu/aim_repo --host 0.0.0.0 --port 43800"
case "$ACTION_NAME" in
  smoke)
    tmux new -d -s trace "source /opt/pytorch/bin/activate && python scripts/smoke_test.py 2>&1 | tee /home/ubuntu/trace/smoke.log"
    ;;
  baseline-matrix)
    python scripts/build_micro_web.py --output ./data/microweb_$RUN_TAG --episodes $N_EPISODES --seed 42
    tmux new -d -s trace "source /opt/pytorch/bin/activate && websleuth baseline-matrix --config $CONFIG --manifest ./data/microweb_$RUN_TAG/manifest.json --aim-repo /home/ubuntu/aim_repo --aim-experiment trace-baseline-matrix --aim-tag run=$RUN_TAG 2>&1 | tee /home/ubuntu/trace/matrix.log"
    ;;
  *)
    python scripts/build_micro_web.py --output ./data/microweb_$RUN_TAG --episodes $N_EPISODES --seed 42
    tmux new -d -s trace "source /opt/pytorch/bin/activate && websleuth stream --config $CONFIG --manifest ./data/microweb_$RUN_TAG/manifest.json --aim-repo /home/ubuntu/aim_repo --aim-experiment trace-continual-350m --aim-tag run=$RUN_TAG 2>&1 | tee /home/ubuntu/trace/stream.log"
    ;;
esac
echo started
REMOTE
}

open_tunnel(){
  local ip="$1"
  log "opening SSH tunnel localhost:43800 → ec2:43800"
  pkill -f "ssh.*-L 43800:localhost:43800.*$ip" 2>/dev/null || true
  ssh $SSH_OPTS -fN -L 43800:localhost:43800 ubuntu@"$ip"
  log "Aim dashboard at http://localhost:43800"
  log "tail the run: ssh -i $KEY_PATH ubuntu@$ip 'tmux capture-pane -p -t trace | tail -50'"
}

fetch_results(){
  local ip="$1"
  mkdir -p "$LOCAL_RESULTS" "$LOCAL_AIM"
  log "rsync runs/ → $LOCAL_RESULTS"
  rsync -az -e "ssh $SSH_OPTS" "ubuntu@$ip:$REMOTE_TRACE/runs/" "$LOCAL_RESULTS/" || true
  log "rsync .aim → $LOCAL_AIM"
  rsync -az -e "ssh $SSH_OPTS" "ubuntu@$ip:$REMOTE_AIM/" "$LOCAL_AIM/" || true
  log "done. local aim repo: aim up --repo $LOCAL_AIM"
}

cmd_launch(){
  local pair; pair=$(launch_instance)
  local iid="${pair%%:*}"
  local ip="${pair##*:}"
  push_code "$ip"
  install_and_run "$ip" "$ACTION"
  open_tunnel "$ip"
  log "IID=$iid IP=$ip"
  log "to fetch results later: IID=$iid $0 fetch"
  log "to terminate later:     IID=$iid $0 terminate"
}

cmd_attach(){
  [[ -n "${IID:-}" ]] || die "set IID=i-..."
  local ip; ip=$(resolve_iid_to_ip "$IID")
  [[ "$ip" != "None" && -n "$ip" ]] || die "no public IP for $IID"
  open_tunnel "$ip"
}

cmd_fetch(){
  [[ -n "${IID:-}" ]] || die "set IID=i-..."
  local ip; ip=$(resolve_iid_to_ip "$IID")
  fetch_results "$ip"
}

cmd_terminate(){
  [[ -n "${IID:-}" ]] || die "set IID=i-..."
  log "terminating $IID"
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$IID" >/dev/null
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$IID"
  log "terminated"
}

case "$ACTION" in
  launch)    cmd_launch ;;
  attach)    cmd_attach ;;
  fetch)     cmd_fetch ;;
  terminate) cmd_terminate ;;
  smoke|stream|baseline-matrix) cmd_launch ;;
  *) die "unknown action: $ACTION" ;;
esac
