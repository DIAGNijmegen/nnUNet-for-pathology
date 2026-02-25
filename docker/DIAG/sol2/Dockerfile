FROM doduo1.umcn.nl/uokbaseimage/diag:latest

RUN echo "PYTHONUNBUFFERED=1" >> /etc/environment && \
    echo "OMP_NUM_THREADS=1" >> /etc/environment

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y \
    libgeos-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install --no-cache-dir \
    numpy\<2 \
    pandas \    
    shapely \
    albumentations==1.2.1 \
    git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main \
    acvl-utils==0.2 \
    geojson \
    ipympl \
    scikit-learn \
    pycm \
    tqdm \
    seaborn \
    carbontracker \
    cryptography \
    paramiko

# Clone and configure nnUNet-for-pathology v1
RUN git clone https://github.com/DIAGNijmegen/nnUNet-for-pathology.git /home/user/nnunet && \
    git -C /home/user/nnunet checkout nnunet_for_pathology_v1 && \
    chown -R user /home/user/nnunet && \
    pip3 install -e /home/user/nnunet

# Copy the wrapper from nnUNet-for-pathology v1's example usage folder
RUN cp /home/user/nnunet/nnunet_for_pathology_example_usage/nnunet_wrapper.py /home/user/nnunet_wrapper.py && \
    chmod +x /home/user/nnunet_wrapper.py && \
    ln -s /home/user/nnunet_wrapper.py /usr/local/bin/nnunet

# Clone and configure nnUNet-for-pathology v2
RUN git clone https://github.com/DIAGNijmegen/nnUNet-for-pathology.git /home/user/nnunet_v2 && \
    git -C /home/user/nnunet_v2 checkout nnunet_for_pathology_v2 && \
    chown -R user /home/user/nnunet_v2 && \
    pip3 install --no-use-pep517 -e /home/user/nnunet_v2
