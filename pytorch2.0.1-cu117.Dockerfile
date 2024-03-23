# https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# NOTE:
# pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime has python version 3.10.8, system version Ubuntu 18.04.6 LTS
# pytorch/pytorch:1.10.1-cuda11.3-cudnn8-runtime has python version 3.7.x
# pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime has python version 3.10.11, system version Ubuntu 20.04.6 LTS
# FROM python/python:3.8-slim

# set the environment variable to avoid interactive installation
# which might stuck the docker build process
ENV DEBIAN_FRONTEND=noninteractive

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

RUN mkdir /fl_sim
COPY ./ /fl_sim
WORKDIR /fl_sim


## Install your dependencies here using apt install, etc.

RUN apt update && apt upgrade -y && apt clean
RUN apt install ffmpeg libsm6 libxext6 tar unzip wget vim nano -y

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements-no-torch.txt
RUN pip install -r requirements-viz.txt
# RUN pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.15.2+cu117 --no-deps -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-optimizer --no-deps
RUN python -m pip cache purge

RUN python -m pip install .

# RUN python docker_test.py
