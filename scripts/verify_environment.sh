#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "${PROJECT_ROOT}/scripts/runtime_env.sh"
cd -- "${PROJECT_ROOT}"
if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "Missing vulkaninfo; install Vulkan loader/tools before training" >&2
  exit 1
fi
VULKAN_SUMMARY="${TMPDIR:-/tmp}/ucf-vulkan-summary.$$"
vulkaninfo --summary >"${VULKAN_SUMMARY}" 2>/dev/null
if ! grep -q "deviceName.*NVIDIA H20" "${VULKAN_SUMMARY}"; then
  rm -f -- "${VULKAN_SUMMARY}"
  echo "Vulkan NVIDIA H20 gate failed" >&2
  exit 1
fi
rm -f -- "${VULKAN_SUMMARY}"
"${UCF_PYTHON_BIN_RESOLVED}" -c '
import torch, lerobot, transformers
assert torch.__version__.startswith("2.8."), torch.__version__
assert torch.cuda.is_available() and torch.cuda.device_count() >= 4
assert lerobot.__version__ == "0.5.1", lerobot.__version__
assert transformers.__version__ == "5.3.0", transformers.__version__
'
"${UCF_PYTHON_BIN_RESOLVED}" -m unittest \
  tests.regression.test_project_boundary \
  tests.integration.test_environment_contract \
  tests.regression.test_config_contract -v
"${UCF_PYTHON_BIN_RESOLVED}" -m unittest \
  tests.integration.test_real_dinov3.RealDinoV3IntegrationTest.test_exact_revision_cuda_state_encoder_and_backward -v
