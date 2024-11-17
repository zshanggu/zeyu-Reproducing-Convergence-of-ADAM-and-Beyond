# Base image with CUDA 11.2, cuDNN 8, and Ubuntu 20.04
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install necessary dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.7 and pip
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3-pip \
    python3.7-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set python3.7 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Upgrade pip to the latest version
RUN python3.7 -m pip install --upgrade pip

# Verify CUDA installation
RUN nvcc --version

# Verify Python and pip versions
RUN python3.7 --version && pip --version

# Update and upgrade system packages
RUN apt update && apt upgrade -y

# Install git
RUN apt-get install -y git

# Install libgl1-mesa-glx
RUN apt-get install -y libgl1-mesa-glx
