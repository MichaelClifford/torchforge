# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Base Dockerfile for TorchForge Kubernetes worker pods.
#
# This image builds on top of the Monarch nightly image which already
# includes torchmonarch, PyTorch, and CUDA. We add vLLM, TorchForge,
# and other dependencies on top.
#
# Why nightly? The stable release (0.3.0) is missing KubernetesJob
# provisioning support (ImageSpec, job.apply(), job.kill()).
#
# Build:
#   docker build -t torchforge-worker:latest -f docker/Dockerfile .
#
# Usage in TorchForge config:
#   provisioner:
#     launcher: k8s
#     k8s_image: "torchforge-worker:latest"
#
# Customize:
#   - Override MONARCH_IMAGE to use a different nightly tag
#   - Add additional dependencies in the final stage

ARG MONARCH_IMAGE=ghcr.io/meta-pytorch/monarch-nightly:0.4.0.dev20260309-cuda12.8

FROM ${MONARCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install git (needed for torchstore from source) and remaining dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir --break-system-packages \
        "torchtitan==0.2.0" \
        "vllm>=0.13.0,<0.14.0" \
        "datasets>=2.21.0" \
        "tokenizers" \
        "omegaconf" \
        "hf_transfer" \
        "six" \
        "setuptools<80" \
        "kubernetes" \
        "pygtrie" \
        "portpicker" \
    && pip install --no-cache-dir --no-deps --break-system-packages \
        "torchstore @ git+https://github.com/meta-pytorch/torchstore.git@batch-put" \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Copy TorchForge source and install (--no-deps to avoid re-resolving
# torch/CUDA versions from the base image).
# The source tree is kept because apps/ (GRPO, SFT entry points) and
# experimental configs are not part of the pip package.
COPY . /workspace/torchforge
RUN pip install --no-cache-dir --no-deps --break-system-packages /workspace/torchforge

WORKDIR /workspace/torchforge

# Enable fast HuggingFace downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Disable RDMA for TorchStore by default.
# Override in your pod spec env if your
# cluster supports RDMA (e.g. SR-IOV + Multus CNI).
ENV TORCHSTORE_RDMA_ENABLED=0

# Default Monarch worker port (matches KubernetesJob default)
EXPOSE 26600

# The Monarch worker runtime is started by the monarch-kubernetes operator.
# This CMD is a fallback for manual testing / debugging.
CMD ["python", "-m", "monarch.worker"]
