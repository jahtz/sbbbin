FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

LABEL org.opencontainers.image.source=https://github.com/jahtz/sbbbin
LABEL org.opencontainers.image.description="Pixelwise binarization with selectional auto-encoders in Keras"
LABEL org.opencontainers.image.licenses=APACHE-2.0

RUN apt-get update && \
    apt-get install -y software-properties-common curl libopencv-dev libeigen3-dev build-essential cmake pkg-config \
    libjpeg-dev libpng-dev libtiff-dev libgtk-3-dev libcanberra-gtk* libdcmtk-dev libgstreamer1.0-dev libv4l-dev \
    libatlas-base-dev gfortran && \
    add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && \
    apt-get install -y build-essential python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python && \
    ln -sf /usr/bin/pip${PYTHON_VERSION} /usr/local/bin/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY sbbbin ./sbbbin
COPY LICENSE .
COPY pyproject.toml .
COPY README.md .
RUN pip${PYTHON_VERSION} install .
RUN sbbbin --version

WORKDIR /data
ENTRYPOINT ["sbbbin"]
