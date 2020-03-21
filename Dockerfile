FROM tensorflow/tensorflow:2.0.0-gpu-py3
COPY . ./Siam-RL
RUN pip install -U pip && pip install -e Siam-RL
