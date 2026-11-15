#!/usr/bin/env bash
# One-shot local setup for WSL Ubuntu (also works on Linux/macOS).
# Checks Docker, installs Python/make if missing, creates the venv and .env.local,
# starts the stack and runs the Phase 1 smoke test. Safe to run again.
#
#   cd /mnt/c/Users/media/Documents/cryptosentinel
#   bash scripts/setup_wsl.sh
set -euo pipefail
cd "$(dirname "$0")/.."

step() { echo; echo "=== $* ==="; }
fail() { echo; echo "STOPPED: $*"; exit 1; }

step "1/4 Docker"
command -v docker >/dev/null 2>&1 || fail "Docker not found. Install Docker Desktop for Windows, then in Docker Desktop: Settings > Resources > WSL integration > turn ON Ubuntu. Then run this script again."
if ! docker info >/dev/null 2>&1; then
  if [ -S /var/run/docker.sock ] && ! id -nG | grep -qw docker; then
    fail "Docker is installed but your user cannot use it. Run: sudo usermod -aG docker \$USER  then close Ubuntu, reopen it, and run this script again."
  fi
  fail "Docker is installed but not running. If you use Docker Desktop: open it, wait until it says 'Engine running', check Settings > Resources > WSL integration > Ubuntu is ON. Then run this script again."
fi
docker compose version >/dev/null 2>&1 || fail "'docker compose' is missing. Update Docker Desktop (or install the docker-compose-plugin package)."
echo "Docker OK (server $(docker version --format '{{.Server.Version}}'))"

step "2/4 Python and make"
need=()
command -v make >/dev/null 2>&1 || need+=(make)
command -v python3 >/dev/null 2>&1 || need+=(python3)
python3 -c "import ensurepip, venv" >/dev/null 2>&1 || need+=(python3-venv python3-pip)
if [ ${#need[@]} -gt 0 ]; then
  echo "Installing: ${need[*]}  (Ubuntu will ask for your password)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq "${need[@]}"
fi
echo "Python OK ($(python3 --version)), make OK"

step "3/4 Python environment and .env.local"
if [ ! -x .venv/bin/pip ]; then
  rm -rf .venv 2>/dev/null || true
  make venv
else
  .venv/bin/pip install -q -r requirements.txt
fi
make env

step "4/4 Start the stack and run the smoke test (first run downloads images: a few minutes)"
make check

echo
echo "All good. Kafka UI: http://localhost:8080   SeaweedFS files: http://localhost:8888"
echo "Stop the stack with: make down"
