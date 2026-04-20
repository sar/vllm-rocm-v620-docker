# syntax=docker/dockerfile:1.4
# =============================================================================
# Dockerfile: vLLM Inference ONLY for AMD V620/W6800 (gfx1030) + ROCm 7.2.1
# Base: rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
# Purpose: Inference serving (NOT fine-tuning)
# =============================================================================

ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE} AS base

# =============================================================================
# Environment Configuration (CRITICAL for gfx1030)
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # ROCm + gfx1030 specific [[5]][[76]]
    HIP_VISIBLE_DEVICES=all \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    BNB_ROCM_ARCH=gfx1030 \
    # vLLM optimizations
    VLLM_USE_TRITON_FLASH_ATTN=0 \
    PYTORCH_TUNABLEOP_ENABLED=0 \
    # Prevent ROCm memory fragmentation
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Hugging Face cache
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

WORKDIR /workspace

# =============================================================================
# Stage 1: Install uv (fast Python package manager) + system deps
# =============================================================================
FROM base AS builder

# Install uv (faster than pip for dependency resolution) [[36]][[40]]
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install system build dependencies for vLLM source compilation
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    rocm-hip-sdk \
    rocm-device-libs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Install Python dependencies + build vLLM from source
# =============================================================================
FROM builder AS vllm-build

# Install core dependencies via uv (much faster than pip)
RUN uv pip install --system --no-cache \
    # Core inference stack
    transformers>=4.51.0 \
    accelerate>=1.2.0 \
    datasets>=3.2.0 \
    # Async I/O for vLLM server
    aiohttp>=3.9.0 \
    uvloop>=0.19.0 \
    fastapi>=0.115.0 \
    uvicorn>=0.32.0 \
    # Utilities
    pyyaml>=6.0 \
    sentencepiece>=0.2.0 \
    tiktoken>=0.8.0

# Clone vLLM at stable version (v0.8.5 has best ROCm 7.2 compatibility) [[26]]
ARG VLLM_VERSION=v0.8.5
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /tmp/vllm-src
WORKDIR /tmp/vllm-src

# =============================================================================
# CRITICAL WORKAROUND: Patch setup.py to force GPU_TARGETS [[27]][[32]]
# This fixes the bug where PYTORCH_ROCM_ARCH doesn't reach CMake
# =============================================================================
RUN sed -i '/cmake_args = \[/a\        cmake_args += ["-DGPU_TARGETS=gfx1030"]' setup.py && \
    grep -q "GPU_TARGETS=gfx1030" setup.py && echo "✅ GPU_TARGETS patch applied" || (echo "❌ Patch failed!" && exit 1)

# Build vLLM from source with gfx1030 target
# BUILD_FA=0 disables Flash Attention (unstable on RDNA2/gfx1030) [[7]]
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy \
    BUILD_FA=0 \
    PYTORCH_ROCM_ARCH=gfx1030 \
    GPU_TARGETS=gfx1030 \
    MAX_JOBS=$(nproc) \
    uv pip install --system --no-cache -e . --verbose 2>&1 | tee /tmp/vllm_build.log

# Verify installation
RUN python -c "import vllm; print(f'✅ vLLM {vllm.__version__} installed')" && \
    python -c "from vllm import _custom_ops; print('✅ vLLM custom ops loaded')"

# =============================================================================
# Stage 3: Final minimal inference image
# =============================================================================
FROM base AS final

# Copy vLLM installation from build stage
COPY --from=vllm-build /usr/local/lib/python3.12/dist-packages/vllm /usr/local/lib/python3.12/dist-packages/vllm
COPY --from=vllm-build /usr/local/bin/vllm /usr/local/bin/vllm

# Create non-root user for security
RUN useradd -m -u 1000 -G video,render vllmuser && \
    mkdir -p /workspace/models /workspace/.cache/huggingface && \
    chown -R vllmuser:vllmuser /workspace

USER vllmuser

# Expose vLLM OpenAI-compatible API port
EXPOSE 8000

# Health check script
COPY <<'EOF' /workspace/scripts/health_check.sh
#!/bin/bash
set -e
echo "🔍 Checking ROCm + vLLM environment..."
python -c "import torch; assert torch.cuda.is_available(), 'ROCm not detected!'"
python -c "import vllm; print(f'✅ vLLM {vllm.__version__} ready')"
python -c "from vllm import _custom_ops; print('✅ Custom ops loaded')"
echo "✅ Health check passed - ready for inference"
EOF

RUN chmod +x /workspace/scripts/health_check.sh

# Default command
CMD ["/bin/bash"]