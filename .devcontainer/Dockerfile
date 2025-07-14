FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install the ffmpeg and libhdf5-dev package
RUN apt-get update && apt-get install -y ffmpeg libhdf5-dev

#COPY requirements.txt /tmp/pip-tmp/
#RUN pip3 install --upgrade pip
#RUN pip3 install --requirement /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp