# stackrl

Implementation of a model-free reinforcement learning approach to dry stacking with irregular rocks. Project developed to support the MSc thesis [From rocks to walls: a machine learning approach for lunar base construction][thesis]. Follow the link for further informations.

### Contents
* [Description](#description)
* [Instalation](#instalation)
* [Usage](#usage)

## Description

The contents of this project are divided in three main sub-packages:
* [`stackrl.envs`][envs] contains the implementation of the simulated environment, along with related utilities;
* [`stackrl.nets`][nets] contains the implementation of the neural networks used as value estimators;
* [`stackrl.agents`][agents] contains the implementation of the reinforcement learnig algorithm (DQN) used to learn.

The class [`stackrl.Training`][training] provides an interface for the training sessions using elements from the above sub-packages, and saves checkpoints and logs for the learning curves. Under [`stackrl.train`][train] you can find other utilities to load learned policies and plot the learing curves.

<sub>[Contents](#contents)</sub>

## Instalation

You can install `stackrl` with:

    git clone https://github.com/menezesandre/stackrl.git
    pip install -e ./stackrl

Alternatively, you can directly [use the Docker image](#usage_docker).

<sub>[Contents](#contents)</sub>

## Usage

Run this package with:

    python -m stackrl <command>

<a name="usage_docker"></a> You can use this package via the [Docker image][container]:

    docker run --rm -u $(id -u):$(id -g) -v $(pwd):\home -v \home  menezesandre/stackrl <command>

<sub>[Contents](#contents)</sub>

[git]: https://github.com/menezesandre/stackrl
[container]: https://hub.docker.com/r/menezesandre/stackrl
[thesis]: https://fenix.tecnico.ulisboa.pt/cursos/meaer/dissertacao/1691203502344087
[envs]: /stackrl/envs
[agents]: /stackrl/agents
[nets]: /stackrl/nets
[train]: /stackrl/train
[training]: /stackrl/train/training.py
