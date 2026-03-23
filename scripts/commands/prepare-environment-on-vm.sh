#!/usr/bin/env bash

set -euo pipefail

usage() {
	echo "Usage: $0 <ip> <path_to_ssh_key> [remote_user] [remote_project_dir] [python_version]"
	echo "Example: $0 203.0.113.10 ~/.ssh/id_gpu brwsx /workspace/hsi 3.11"
	echo "Optional env: SSH_PORT=2222"
}

if [[ "$#" -lt 2 || "$#" -gt 5 ]]; then
	echo "Invalid number of arguments."
	usage
	exit 1
fi

IP="$1"
PATH_TO_KEY="$2"
REMOTE_USER="${3:-$USER}"
REMOTE_PROJECT_DIR="${4:-/workspace/hsi}"
REQUESTED_PYTHON_VERSION="${5:-}"
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
	exit 1
fi

if ! [[ "${SSH_PORT}" =~ ^[0-9]+$ ]] || [[ "${SSH_PORT}" -lt 1 || "${SSH_PORT}" -gt 65535 ]]; then
	echo "Invalid SSH_PORT: ${SSH_PORT}"
	exit 1
fi

if [[ -z "${REMOTE_PROJECT_DIR}" || "${REMOTE_PROJECT_DIR}" == "/" ]]; then
	echo "Refusing to run with an empty project dir or '/'."
	exit 1
fi

echo "Preparing Python environment on ${REMOTE_USER}@${IP}:${REMOTE_PROJECT_DIR}..."

ssh -i "${PATH_TO_KEY}" -p "${SSH_PORT}" "${REMOTE_USER}@${IP}" \
	"PROJECT_DIR='${REMOTE_PROJECT_DIR}' REQUESTED_PY='${REQUESTED_PYTHON_VERSION}' REPO_URL='${REPO_URL:-}' bash -se" << 'EOF'
set -euo pipefail

require_cmd() {
	local cmd="$1"
	if ! command -v "${cmd}" >/dev/null 2>&1; then
		echo "Missing required command: ${cmd}"
		exit 1
	fi
}

install_python_with_apt() {
	local py_ver="$1"
	local apt_py="python${py_ver}"
	local distutils_pkg="${apt_py}-distutils"

	if command -v "${apt_py}" >/dev/null 2>&1; then
		echo "${apt_py} already installed."
		return 0
	fi

	require_cmd apt-get
	echo "Installing ${apt_py} and venv support..."
	apt_install update
	apt_install install -y "${apt_py}" "${apt_py}-venv" || {
		echo "Direct apt install failed. Attempting deadsnakes PPA fallback..."
		apt_install install -y software-properties-common
		apt_add_repo -y ppa:deadsnakes/ppa
		apt_install update
		apt_install install -y "${apt_py}" "${apt_py}-venv"
	}

	if apt-cache show "${distutils_pkg}" >/dev/null 2>&1; then
		apt_install install -y "${distutils_pkg}"
	else
		echo "Skipping optional package ${distutils_pkg} (not available for ${apt_py})."
	fi
}

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR not set}"
REQUESTED_PY="${REQUESTED_PY:-}"
REPO_URL="${REPO_URL:-}"

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

apt_install() {
	as_root apt-get "$@"
}

apt_add_repo() {
	as_root add-apt-repository "$@"
}

if [[ ! -d "${PROJECT_DIR}" ]]; then
	echo "Remote project directory not found. Creating: ${PROJECT_DIR}"
	if ! mkdir -p "${PROJECT_DIR}" 2>/dev/null; then
		as_root mkdir -p "${PROJECT_DIR}"
	fi
fi

cd "${PROJECT_DIR}"

if [[ ! -d .git ]]; then
	if [[ -z "$(ls -A . 2>/dev/null || true)" ]]; then
		if [[ -z "${REPO_URL}" ]]; then
			echo "REPO_URL is required to clone into an empty project directory."
			exit 1
		fi
		require_cmd git
		echo "Project directory is empty. Cloning repository from ${REPO_URL}..."
		git clone "${REPO_URL}" .
	else
		echo "Project directory exists but is not a git repository. Skipping clone to avoid overwriting files."
	fi
fi

if [[ -n "${REQUESTED_PY}" ]]; then
	PYTHON_VERSION="${REQUESTED_PY}"
	echo "Using user-requested Python version: ${PYTHON_VERSION}"
else
	PYTHON_VERSION="3.11"
	echo "No --python-version provided. Defaulting to Python ${PYTHON_VERSION}."
fi

install_python_with_apt "${PYTHON_VERSION}"

PYTHON_BIN="python${PYTHON_VERSION}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
	echo "Expected interpreter not found after install: ${PYTHON_BIN}"
	exit 1
fi

echo "Creating/updating virtual environment at ${PROJECT_DIR}/.venv"
"${PYTHON_BIN}" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

echo "Environment ready."
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
EOF

echo "Remote python environment preparation completed successfully."