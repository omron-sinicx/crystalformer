FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Setup proxies if needed
# ENV http_proxy 'http://internal-182-vpn-elb-1434222708.ap-northeast-1.elb.amazonaws.com:8080/'
# ENV http_proxy $http_proxy
# ENV HTTP_PROXY $http_proxy
# ENV https_proxy $http_proxy
# ENV HTTPS_PROXY $http_proxy
# ENV ftp_proxy $http_proxy
# ENV FTP_PROXY $http_proxy

# Install basics
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
 && apt-get install -y software-properties-common apt-utils git wget curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get clean

# Fix the issue of missing GLIBCXX_3.4.29.
# The 'software-properties-common' is installed above to run add-apt-repository.
RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
 && apt update -y \
 && apt upgrade -y libstdc++6 \
 && apt-get clean

# Install Miniconda
ARG PYTHON_VERSION=3.11
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install -y numpy scipy matplotlib \
 && conda clean -i -t -y
RUN pip install --no-cache-dir pymatgen

ARG TORCH=2.1.1
ARG TORCH_PYG=2.1.0
ARG CUDA=cu121

RUN pip install --no-cache-dir \
    torch==${TORCH}+${CUDA} \
    --extra-index-url https://download.pytorch.org/whl/${CUDA}

RUN pip install --no-cache-dir \
    pyg_lib \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    torch_geometric -f https://data.pyg.org/whl/torch-${TORCH_PYG}+${CUDA}.html

RUN pip install --no-cache-dir \
    pytorch-lightning==2.1.3 \
    cupy-cuda12x pytorch-pfn-extras \
    jarvis-tools \
    tensorboard

ENV CUDA_DEVICE_ORDER PCI_BUS_ID
