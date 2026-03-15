#!/usr/bin/env bash

set -euo pipefail

usage() {
	echo "Usage: $0 <ip> <path_to_ssh_key> <path_to_dataset> [remote_user]"
	echo "Example: $0 203.0.113.10 ~/.ssh/id_gpu ~/datasets/hsi brwsx"
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

DATASET_SIZE_HUMAN="$(du -sh "${DATASET_SOURCE}" | awk '{print $1}')"
echo "Dataset size: ${DATASET_SIZE_HUMAN}"

echo "Preparing remote /workspace directory on ${REMOTE_USER}@${IP}..."
ssh -i "${PATH_TO_KEY}" "${REMOTE_USER}@${IP}" \
	"sudo mkdir -p /workspace/data && sudo chmod --recursive 777 /workspace"

echo "Syncing dataset directly to ${REMOTE_USER}@${IP}:/workspace/data/..."
rsync -a --partial --append-verify --info=progress2 -e "ssh -i ${PATH_TO_KEY}" "${DATASET_SOURCE}" "${REMOTE_USER}@${IP}:/workspace/data/"

echo "Configuring remote repository and DATASET_ROOT..."
ssh -i "${PATH_TO_KEY}" "${REMOTE_USER}@${IP}" << 'EOF'
set -euo pipefail

cd /workspace

if [[ ! -d hsi ]]; then
	git clone https://github.com/mhx1467/master-thesis-code hsi
else
	echo "Repository /workspace/hsi already exists. Skipping clone."
fi

DATASET_ROOT=/workspace/data/hyspectnet-11k/hyspecnet-11k-full/
export DATASET_ROOT

persist_env_var() {
	local target_file="$1"
	local export_line='export DATASET_ROOT=/workspace/data/hyspectnet-11k/hyspecnet-11k-full/'

	touch "${target_file}"
	if ! grep -Fq "${export_line}" "${target_file}"; then
		echo "${export_line}" >> "${target_file}"
	fi
}

persist_env_var "$HOME/.profile"
persist_env_var "$HOME/.bashrc"
persist_env_var "$HOME/.zshrc"

cat > /workspace/hsi/.env << 'ENVVARS'
DATASET_ROOT=/workspace/data/hyspectnet-11k/hyspecnet-11k-full/
ENVVARS

echo "Wrote DATASET_ROOT to /workspace/hsi/.env and persisted it in shell startup files"
EOF

echo "Dataset sync completed."
