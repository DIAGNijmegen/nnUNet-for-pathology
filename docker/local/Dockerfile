ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8

# ------------------------------------------------------------
# FROM
# ------------------------------------------------------------
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

# ------------------------------------------------------------
# 1. System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libcrypto++-dev \
    libgomp1 \
    libopenblas0-openmp \
    libvips-dev \
    openslide-tools \
    python3 \
    python3-pip \
    git \
    wget \
    openssh-server \
    libdcmtk16 \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libqt5concurrent5 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. ASAP installation
# ------------------------------------------------------------
RUN mkdir -p /tmp/asap && \
    wget -q "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2204.deb" \
        -O /tmp/asap/asap.deb && \
    apt-get update && \
    apt-get install -y /tmp/asap/asap.deb && \
    rm -rf /tmp/asap && \
    ldconfig && \
    SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    echo "/opt/ASAP/bin/" > "${SITE_PACKAGES}/asap.pth" && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 3. Python dependencies
# ------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    numpy\<2 \
    pandas \
    shapely \
    albumentations==1.2.1 \
    acvl-utils==0.2 \
    geojson \
    ipympl \
    scikit-learn \
    pycm \
    tqdm \
    seaborn \
    cryptography \
    paramiko \
    openpyxl \
    wandb \
    flask \
    scikit-survival \
    lifelines \
    roc-utils

RUN python3 -m pip install --no-cache-dir \
    git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main

# ------------------------------------------------------------
# 4. User setup
# ------------------------------------------------------------
RUN useradd -ms /bin/bash user && \
    (echo user ; echo user) | passwd user && \
    gpasswd -a user sudo && \
    mkdir -p /var/run/sshd

# ------------------------------------------------------------
# 5. nnUNet-for-pathology v1 (still as root)
# ------------------------------------------------------------
WORKDIR /home/user

RUN git clone --depth 1 --branch nnunet_for_pathology_v1 \
    https://github.com/DIAGNijmegen/nnUNet-for-pathology.git nnunet && \
    pip3 install --no-cache-dir -e nnunet

# Install wrapper system-wide BEFORE switching User
RUN cp nnunet/nnunet_for_pathology_example_usage/nnunet_wrapper.py /home/user/nnunet_wrapper.py && \
    sed -i '1s|^.*$|#!/usr/bin/python3|' /home/user/nnunet_wrapper.py && \
    chmod +x /home/user/nnunet_wrapper.py && \
    ln -s /home/user/nnunet_wrapper.py /usr/local/bin/nnunet

# ------------------------------------------------------------
# 6. nnUNet-for-pathology v2
# ------------------------------------------------------------
RUN git clone --depth 1 --branch nnunet_for_pathology_v2 \
    https://github.com/DIAGNijmegen/nnUNet-for-pathology.git nnunet_v2 && \
    pip3 install --no-cache-dir -e nnunet_v2

# ------------------------------------------------------------
# 7. Switch to non-root user
# ------------------------------------------------------------
USER user
WORKDIR /home/user

# ------------------------------------------------------------
# 8. Final setup
# ------------------------------------------------------------
ENTRYPOINT ["/bin/bash"]
