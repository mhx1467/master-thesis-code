#!/usr/bin/env bash

set -euo pipefail

usage() {
	echo "Usage: $0 <ip> <path_to_ssh_key> <path_to_dataset> [remote_user]"
	echo "Example: $0 203.0.113.10 ~/.ssh/id_gpu ~/datasets/hsi brwsx"
	echo "Optional env: SSH_PORT=2222"
}

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
	echo "Invalid number of arguments."
	usage
	exit 1
fi

IP="$1"
PATH_TO_KEY="$2"
PATH_TO_DATASET="$3"
REMOTE_USER="${4:-$USER}"
SSH_PORT="${SSH_PORT:-22}"

if [[ -z "${IP}" ]]; then
	echo "IP address is required."
	usage
	exit 1
fi

if [[ -z "${PATH_TO_KEY}" ]]; then
	echo "SSH key path is required."
	usage
	exit 1
fi

if [[ ! -f "${PATH_TO_KEY}" ]]; then
	echo "SSH key file not found: ${PATH_TO_KEY}"
	usage
	exit 1
fi

if [[ -z "${PATH_TO_DATASET}" ]]; then
	echo "Dataset path is required."
	usage
	exit 1
fi

if ! [[ "${SSH_PORT}" =~ ^[0-9]+$ ]] || [[ "${SSH_PORT}" -lt 1 || "${SSH_PORT}" -gt 65535 ]]; then
	echo "Invalid SSH_PORT: ${SSH_PORT}"
	exit 1
fi

if [[ ! -e "${PATH_TO_DATASET}" ]]; then
	echo "Dataset path not found: ${PATH_TO_DATASET}"
	usage
	exit 1
fi

if [[ "${PATH_TO_DATASET}" == "/" ]]; then
	echo "Refusing to transfer root directory '/'."
	exit 1
fi

DATASET_SOURCE="${PATH_TO_DATASET%/}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_ENV_FILE="${LOCAL_PROJECT_ROOT}/.env"

DATASET_SIZE_HUMAN="$(du -sh "${DATASET_SOURCE}" | awk '{print $1}')"
echo "Dataset size: ${DATASET_SIZE_HUMAN}"

echo "Preparing remote /workspace directory on ${REMOTE_USER}@${IP}..."
ssh -i "${PATH_TO_KEY}" -p "${SSH_PORT}" "${REMOTE_USER}@${IP}" << 'EOF'
set -euo pipefail

as_root() {
	if [[ "$(id -u)" -eq 0 ]]; then
		"$@"
		return
	fi

	if command -v sudo >/dev/null 2>&1; then
		sudo "$@"
		return
	fi

	echo "Need root privileges to run: $*"
	exit 1
}

as_root mkdir -p /workspace/data
as_root chmod --recursive 777 /workspace
EOF

echo "Syncing dataset directly to ${REMOTE_USER}@${IP}:/workspace/data/..."
rsync -a --partial --append-verify --info=progress2 -e "ssh -i ${PATH_TO_KEY} -p ${SSH_PORT}" "${DATASET_SOURCE}" "${REMOTE_USER}@${IP}:/workspace/data/"

echo "Configuring remote repository..."
ssh -i "${PATH_TO_KEY}" -p "${SSH_PORT}" "${REMOTE_USER}@${IP}" << 'EOF'
set -euo pipefail

REPO_URL="${REPO_URL:-}"

cd /workspace

if [[ ! -d hsi ]]; then
	if [[ -z "${REPO_URL}" ]]; then
		echo "REPO_URL is required to clone /workspace/hsi when it does not exist."
		exit 1
	fi
	git clone "${REPO_URL}" hsi
else
	echo "Repository /workspace/hsi already exists. Skipping clone."
fi
EOF

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
	echo "Copying local .env to remote /workspace/hsi/.env..."
	rsync -a --info=progress2 -e "ssh -i ${PATH_TO_KEY} -p ${SSH_PORT}" "${LOCAL_ENV_FILE}" "${REMOTE_USER}@${IP}:/workspace/hsi/.env"
else
	echo "No local .env found at ${LOCAL_ENV_FILE}. Remote .env will be created if missing."
fi

echo "Configuring remote environment variables..."
ssh -i "${PATH_TO_KEY}" -p "${SSH_PORT}" "${REMOTE_USER}@${IP}" << 'EOF'
set -euo pipefail

cd /workspace/hsi

if [[ ! -f .env ]]; then
	cat > .env << 'ENVVARS'
DATASET_ROOT=/workspace/data/hyspectnet-11k/hyspecnet-11k-full/
ENVVARS
fi

if ! grep -Eq '^DATASET_ROOT=' .env; then
	echo 'DATASET_ROOT=/workspace/data/hyspectnet-11k/hyspecnet-11k-full/' >> .env
fi

chmod 600 .env

persist_env_loader() {
	local target_file="$1"

	touch "${target_file}"
	if ! grep -Fq '# hsi-env-loader' "${target_file}"; then
		cat >> "${target_file}" << 'ENVLOADER'
# hsi-env-loader
if [ -f /workspace/hsi/.env ]; then
  set -a
  . /workspace/hsi/.env
  set +a
fi
ENVLOADER
	fi
}

persist_env_loader "$HOME/.profile"
persist_env_loader "$HOME/.bashrc"
persist_env_loader "$HOME/.zshrc"

echo "Remote .env secured and shell startup files configured to auto-load it"
EOF

echo "Dataset sync completed."
