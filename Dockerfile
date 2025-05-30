# Use NVIDIA CUDA runtime image instead of devel
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu20.04

# Set the working directory in the container

# Copy the requirements file to the working directory

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends -y python3.10 python3.10-dev python3.10-venv python3.10-distutils

RUN python3.10 -m ensurepip && \
    python3.10 -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

COPY asr/requirements_asr.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the application code
COPY asr/asr_api.py /app/asr_api.py

# Command to run the application
CMD ["python3", "asr_api.py"]