ARG TF_VERSION=latest
# PY3 must be set explicitely for TF_VERSION earlier than 2.1.0
ARG PY3
FROM tensorflow/tensorflow:${TF_VERSION}}-gpu${PY3:+-py3}
COPY . ./Siam-RL
RUN pip install -U pip && pip install -e Siam-RL && ln -s /Siam-RL/apps/train.py /usr/local/bin/train