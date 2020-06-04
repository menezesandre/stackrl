ARG TF_VERSION=latest
FROM tensorflow/tensorflow:${TF_VERSION}}-gpu${TF_VERSION<2.1.0:+-py3}
COPY . ./Siam-RL
RUN pip install -U pip && pip install -e Siam-RL && ln -s /Siam-RL/apps/train.py /usr/local/bin/train