# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM python:3.10-slim

## The MAINTAINER instruction sets the author field of the generated images.
LABEL maintainer="wenh06@gmail.com"

RUN mkdir /fl_sim
COPY ./ /fl_sim
WORKDIR /fl_sim


## Install your dependencies here using apt install, etc.

RUN apt update && apt upgrade -y && apt clean
RUN apt install ffmpeg libsm6 libxext6 tar unzip wget vim nano tk -y

# RUN apt install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# RUN pip install --upgrade pip

# http://mirrors.aliyun.com/pypi/simple/
# http://pypi.douban.com/simple/
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
RUN pip install -r requirements-viz.txt
RUN python -m pip cache purge

RUN python -m pip install .


# RUN python docker_test.py
