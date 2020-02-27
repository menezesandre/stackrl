FROM tensorflow/tensorflow:nightly-gpu
COPY . ./Siam-RL
RUN pip install -U pip && pip install -e Siam-RL
