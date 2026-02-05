# Start with the specified stable PyTorch base image
ARG PYTORCH="2.3.1"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install basic Python packages
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install Cython
RUN pip install --no-cache-dir cython==3.0.8

# Install MMCV from source (as in the original file)
RUN pip install --no-cache-dir --trusted-host download.openmmlab.com mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html

# Install OpenMMLab packages using MIM
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmdet>=3.0.0"

# Install pycocotools
RUN pip install --no-cache-dir pycocotools

# Install MMPose
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN pip install -r requirements.txt && pip install --no-cache-dir --no-build-isolation -v -e .

# Install MMAction2
RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
WORKDIR /mmaction2
RUN pip install --no-build-isolation -v -e .

# Install additional useful packages
RUN pip install --no-cache-dir \
    opencv-python-headless \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    tqdm \
    openpyxl

# --- Runtime Error Patch ---
# This step replicates the manual fix applied to solve runtime errors.
# It uninstalls the potentially incompatible default PyTorch and MMCV versions
# and installs the specific nightly build that is known to work.
RUN pip uninstall torch torchvision torchaudio mmcv mmcv-full -y && \
    pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install mmcv==2.1.0 && \
	pip install importlib_metadata

# Set the final working directory for your projects
WORKDIR /workspace
