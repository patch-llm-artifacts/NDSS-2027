# Base image with CUDA 12.9 and Ubuntu 22.04
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

# Set work directory
WORKDIR /app

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git curl gcc g++ build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Symlink python3 and pip3 for convenience
RUN [ ! -e /usr/bin/python ] && ln -s /usr/bin/python3 /usr/bin/python || true
RUN [ ! -e /usr/bin/pip ] && ln -s /usr/bin/pip3 /usr/bin/pip || true

# Copy requirements.txt and install Python packages
COPY requirements.txt .

# Install PyTorch and TorchVision for CUDA 12.9
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129 && \
    pip install -r requirements.txt

# Install Semgrep for static code analysis
RUN pip install semgrep

# Install PyTorch Geometric (PyG) matching torch version (e.g., 2.3.0)
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.3.0+cu129.html

# Uninstall incompatible torch-scatter (if any) and build from source
RUN pip uninstall -y torch-scatter && \
    pip install git+https://github.com/rusty1s/pytorch_scatter.git


# Copy source code
COPY src/ /app/src/

# Set default command
CMD ["/bin/bash"]
