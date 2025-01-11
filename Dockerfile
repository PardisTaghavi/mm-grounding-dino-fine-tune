FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN pip install openmim
RUN mim install mmengine 
RUN mim install "mmcv>=2.0.0"
#RUN mim install mmdet
