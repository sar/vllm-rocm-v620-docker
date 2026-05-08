# syntax=docker/dockerfile:1.4
# =============================================================================
# vLLM v0.20.0 on AMD gfx1030 (Radeon Pro V620) — ROCm 7.2.1
# =============================================================================

ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE} AS base

# =============================================================================
# Environment Configuration
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HIP_VISIBLE_DEVICES=all \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

WORKDIR /workspace

# =============================================================================
# Stage 1: System build dependencies
# =============================================================================
FROM base AS builder

# ONLY install what the base image doesn't have. 
# DO NOT reinstall rocm-hip-sdk or rocm-device-libs.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Install deps + compile vLLM (Using base's existing /opt/venv)
# =============================================================================
FROM builder AS vllm-build

# The base image already activates /opt/venv via ENV PATH.
# We do NOT create a new venv. We install directly into the base's venv.
# Because the base's venv already has ROCm PyTorch, pip will see it and 
# will NOT overwrite it with the CUDA version.

# Pre-install vLLM build dependencies required because we use --no-build-isolation
RUN pip install setuptools_scm wheel ninja

# Pre-install vLLM's Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
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

# Clone vLLM v0.20.0 to /workspace instead of /tmp
ARG VLLM_VERSION=v0.20.0
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /workspace/vllm-src
WORKDIR /workspace/vllm-src

# Build vLLM.
# --no-build-isolation uses the base image's PyTorch ROCm headers.
# Log is saved to /workspace so we can keep it in the final image.
RUN --mount=type=cache,target=/root/.cache/pip \
    GPU_TARGETS=gfx1030 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    BUILD_FA=0 \
    VLLM_FLASH_ATTN=0 \
    VLLM_FLASHINFER=0 \
    MAX_JOBS=$(nproc) \
    pip install --no-cache-dir --no-build-isolation -v . 2>&1 | tee /workspace/vllm_build.log

# Verify build succeeded. 
# CRITICAL: Run from '/' so it checks the INSTALLED package in site-packages,
# not the local /workspace/vllm-src folder which will false-positive.
RUN cd / && python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" || { \
    echo "❌ Build failed. Compiler errors:"; \
    grep -i "error\|fatal\|hipcc: error" /workspace/vllm_build.log | tail -30; \
    exit 1; }

# =============================================================================
# Stage 3: Final minimal inference image
# =============================================================================
FROM base AS final

# Copy the modified venv from builder with vLLM installed
COPY --from=vllm-build /opt/venv /opt/venv

# Copy the vLLM source and build log into the image for debugging/reference
COPY --from=vllm-build /workspace/vllm-src /workspace/vllm-src
COPY --from=vllm-build /workspace/vllm_build.log /workspace/vllm_build.log

# Set runtime environment
ENV HIP_VISIBLE_DEVICES=all \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    VLLM_TARGET_DEVICE=rocm \
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/.cache/huggingface

EXPOSE 8000

# Default command
CMD ["python", "-m", "vllm.entrypoints.openai.api_server"]