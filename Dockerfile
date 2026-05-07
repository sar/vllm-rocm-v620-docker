# syntax=docker/dockerfile:1.4
# =============================================================================
# vLLM on AMD gfx1030 (Radeon Pro V620) — ROCm 7.2.1
# =============================================================================

ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HIP_VISIBLE_DEVICES=all \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    # CRITICAL: Disable flash-attn and triton flash (unsupported on RDNA2)
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    VLLM_FLASH_ATTN=0 \
    BUILD_FA=0 \
    PYTORCH_TUNABLEOP_ENABLED=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

WORKDIR /workspace

# =============================================================================
# Stage 1: System build dependencies
# =============================================================================
FROM base AS builder

# DO NOT reinstall rocm-hip-sdk. The base image already has it. 
# Reinstalling causes apt repo conflicts and library mismatches.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Python dependencies + compile vLLM
# =============================================================================
FROM builder AS vllm-build

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv venv /opt/venv --system-site-packages
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Pre-install deps using uv (this is fine, uv is great for pure Python)
RUN uv pip install --no-cache \
    "transformers>=4.53.0" \
    "ray[default]>=2.40.0" \
    "pydantic>=2.10.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "aiohttp>=3.11.0" \
    "tiktoken>=0.9.0" \
    "sentencepiece>=0.2.0" \
    "prometheus-client>=0.21.0" \
    "huggingface-hub>=0.28.0" \
    "tokenizers>=0.21.0" \
    "numpy>=1.26,<2.1" \
    "xgrammar>=0.1.10" \
    "lm-format-enforcer>=0.10.0"

ARG VLLM_VERSION=v0.20.0
RUN mkdir -p /workspace && \
    git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /workspace/vllm-src
WORKDIR /workspace/vllm-src

# ---- Build vLLM ----
# CRITICAL CHANGES:
# 1. Switched from `uv pip install` to `pip install`. If the C++ build fails, 
#    `pip` will just fail with the actual compiler error, whereas `uv` falls 
#    back to PyPI, grabs the CUDA wheel, and throws httplib errors.
# 2. Added --no-build-isolation so it uses the ROCm PyTorch headers from the 
#    base image instead of trying to download new ones.
# 3. Added VLLM_FLASHINFER=0 as FlashInfer does not compile on RDNA2.
RUN GPU_TARGETS=gfx1030 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    BUILD_FA=0 \
    VLLM_FLASH_ATTN=0 \
    VLLM_FLASHINFER=0 \
    MAX_JOBS=$(nproc) \
    pip install --no-cache-dir --no-build-isolation -v . 2>&1 | tee /workspace/vllm_build.log

# Verify
RUN python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" || { \
    echo "❌ Build failed. Real compiler error above or in log:"; \
    grep -i "error\|fatal\|hipcc: error" /workspace/vllm_build.log | tail -30; \
    exit 1; }

# =============================================================================
# Stage 3: Final minimal inference image
# =============================================================================
FROM base AS final

COPY --from=vllm-build /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HIP_VISIBLE_DEVICES=all \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0

RUN useradd -m -u 1000 -G video,render vllmuser && \
    mkdir -p /workspace/models /workspace/.cache/huggingface /workspace/scripts && \
    chown -R vllmuser:vllmuser /workspace

USER vllmuser
EXPOSE 8000

COPY <<'EOF' /workspace/scripts/health_check.sh
#!/bin/bash
set -e
echo "🔍 Checking ROCm + vLLM environment..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__} (ROCm: {torch.version.hip})')"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"
python -c "import vllm; print(f'✅ vLLM {vllm.__version__} ready')"
echo "✅ Health check passed"
EOF
RUN chmod +x /workspace/scripts/health_check.sh

CMD ["/workspace/scripts/health_check.sh"]