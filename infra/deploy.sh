#!/bin/bash
# Deploy audio-enhancer training to GCP.
# Usage: ./infra/deploy.sh [upload-data|apply|destroy|ssh-config]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INFRA_DIR="$SCRIPT_DIR"

cd "$INFRA_DIR"

# Load project ID from tfvars
if [ ! -f terraform.tfvars ]; then
  echo "ERROR: infra/terraform.tfvars not found."
  echo "Copy terraform.tfvars.example to terraform.tfvars and fill in your GCP project ID."
  exit 1
fi

PROJECT_ID=$(grep 'project_id' terraform.tfvars | sed 's/.*=\s*"\(.*\)"/\1/')
BUCKET="${PROJECT_ID}-audio-enhancer-data"

case "${1:-help}" in

  upload-data)
    echo "=== Uploading training code to gs://$BUCKET/code/ ==="
    gsutil -m rsync -r \
      -x '(datasets/|checkpoints/|\.venv/|__pycache__/|\.git/|infra/|.*\.log)' \
      "$PROJECT_DIR/" "gs://$BUCKET/code/"

    echo "=== Uploading dataset to gs://$BUCKET/datasets/ ==="
    echo "This will upload ~58GB. Ctrl+C to cancel."
    gsutil -m rsync -r \
      "$PROJECT_DIR/datasets/phase0_combined/" "gs://$BUCKET/datasets/"
    echo "Done!"
    ;;

  apply)
    echo "=== Initializing Terraform ==="
    terraform init

    echo "=== Planning ==="
    terraform plan

    echo ""
    read -p "Apply? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      terraform apply -auto-approve
      echo ""
      echo "=== Deployment complete ==="
      terraform output
      echo ""
      echo "Run: ./infra/deploy.sh ssh-config  to add to your SSH config"
    fi
    ;;

  destroy)
    echo "=== Destroying infrastructure ==="
    terraform destroy
    ;;

  ssh-config)
    IP=$(terraform output -raw vm_external_ip 2>/dev/null)
    if [ -z "$IP" ]; then
      echo "ERROR: VM not deployed yet. Run './infra/deploy.sh apply' first."
      exit 1
    fi

    SSH_CONFIG="$HOME/.ssh/config"
    # Remove old entry if present
    if grep -q "Host audio-enhancer-gcp" "$SSH_CONFIG" 2>/dev/null; then
      sed -i '/Host audio-enhancer-gcp/,/^$/d' "$SSH_CONFIG"
    fi

    cat >> "$SSH_CONFIG" <<EOF

Host audio-enhancer-gcp
  HostName $IP
  User stas
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
EOF

    echo "Added 'audio-enhancer-gcp' to $SSH_CONFIG"
    echo "Open VSCode -> Remote SSH -> Connect to Host -> audio-enhancer-gcp"
    ;;

  status)
    IP=$(terraform output -raw vm_external_ip 2>/dev/null)
    echo "VM IP: $IP"
    echo "Bucket: gs://$BUCKET"
    echo ""
    echo "SSH: ssh stas@$IP"
    echo "VSCode: Remote SSH -> audio-enhancer-gcp"
    ;;

  help|*)
    echo "Usage: ./infra/deploy.sh <command>"
    echo ""
    echo "Commands:"
    echo "  apply        - Deploy GCP infrastructure (VM + GCS bucket)"
    echo "  upload-data  - Upload code + dataset to GCS bucket"
    echo "  ssh-config   - Add VM to ~/.ssh/config for VSCode Remote SSH"
    echo "  status       - Show deployment info"
    echo "  destroy      - Tear down all GCP resources"
    ;;
esac
