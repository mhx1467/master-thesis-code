#!/usr/bin/env bash

set -euo pipefail

usage() {
	echo "Usage: $0 <ip> <path_to_ssh_key> [remote_user] [remote_project_dir] [python_version]"
	echo "Example: $0 203.0.113.10 ~/.ssh/id_gpu brwsx /workspace/hsi 3.10"
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
	"PROJECT_DIR='${REMOTE_PROJECT_DIR}' REQUESTED_PY='${REQUESTED_PYTHON_VERSION}' bash -se" << 'EOF'
set -euo pipefail

require_cmd() {
	local cmd="$1"
	if ! command -v "${cmd}" >/dev/null 2>&1; then
		echo "Missing required command: ${cmd}"
		exit 1
	fi
}

infer_python_from_pyproject() {
	local pyproject_file="$1"

	if [[ ! -f "${pyproject_file}" ]]; then
		return 1
	fi

	local req_line
	req_line="$(grep -E '^requires-python\s*=\s*"' "${pyproject_file}" || true)"
	if [[ -z "${req_line}" ]]; then
		return 1
	fi

	# Extract the first X.Y from constraints like ">=3.10,<3.13".
	local inferred
	inferred="$(printf '%s' "${req_line}" | grep -Eo '[0-9]+\.[0-9]+' | head -n 1 || true)"
	if [[ -z "${inferred}" ]]; then
		return 1
	fi

	printf '%s\n' "${inferred}"
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
	sudo apt-get update
	sudo apt-get install -y "${apt_py}" "${apt_py}-venv" || {
		echo "Direct apt install failed. Attempting deadsnakes PPA fallback..."
		sudo apt-get install -y software-properties-common
		sudo add-apt-repository -y ppa:deadsnakes/ppa
		sudo apt-get update
		sudo apt-get install -y "${apt_py}" "${apt_py}-venv"
	}

	if apt-cache show "${distutils_pkg}" >/dev/null 2>&1; then
		sudo apt-get install -y "${distutils_pkg}"
	else
		echo "Skipping optional package ${distutils_pkg} (not available for ${apt_py})."
	fi
}

PROJECT_DIR="${PROJECT_DIR:?PROJECT_DIR not set}"
REQUESTED_PY="${REQUESTED_PY:-}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
	echo "Remote project directory not found: ${PROJECT_DIR}"
	exit 1
fi

cd "${PROJECT_DIR}"

if [[ -n "${REQUESTED_PY}" ]]; then
	PYTHON_VERSION="${REQUESTED_PY}"
	echo "Using user-requested Python version: ${PYTHON_VERSION}"
else
	PYTHON_VERSION="$(infer_python_from_pyproject pyproject.toml || true)"
	if [[ -z "${PYTHON_VERSION}" ]]; then
		PYTHON_VERSION="3.10"
		echo "Could not infer Python version from pyproject.toml. Falling back to ${PYTHON_VERSION}."
	else
		echo "Inferred Python version from pyproject.toml: ${PYTHON_VERSION}"
	fi
fi

if ! command -v sudo >/dev/null 2>&1; then
	echo "sudo is required to install Python packages on the VM."
	exit 1
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