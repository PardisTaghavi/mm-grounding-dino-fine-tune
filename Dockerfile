FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install the required packages
RUN apt-get update \
    && apt-get install -y git ninja-build emacs-nox \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies separately to manage memory better
RUN pip install --no-cache-dir openmim opencv-python-headless==4.8.1.78

# Install MMEngine
RUN mim install --no-cache-dir mmengine

# Install specific version of MMCV with CUDA support
RUN pip install --no-cache-dir mmcv==2.0.1

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .

WORKDIR /mmdetection
