# stackrl

A model-free reinforcement learning approach to dry stacking with irregular blocks. [Source][src]

### Contents
* [Description](#description)
* [Instalation](#instalation)
* [Usage](#usage)

## [Description](#contents)

This package contains the implementation of:
* an [environment][env] that simulates an abstraction of the task.
* the [reinforcement learning algorithm][agents] used to learn the task.
* the [neural network][nets] used to estimate the action values.

## [Instalation](#contents)

You can install `stackrl` with:

    git clone https://github.com/menezesandre/stackrl.git
    pip install -e ./stackrl

Alternatively, you can [directly use the Docker image](#docker).

## [Usage](#contents)

Run this package with:

    python -m stackrl <command>

<a name="docker"></a> You can run this package using the Docker image:

    docker run [--gpus=all] --rm -u $(id -u):$(id -g) -v $(pwd):\home -v \home  menezesandre/stackrl python [-m stackrl <command>]

[src]: https://github.com/menezesandre/stackrl
[thesis]: https://fenix.tecnico.ulisboa.pt/cursos/meaer/dissertacao/1691203502344087
[envs]: /stackrl/envs
[agents]: /stackrl/agents
[nets]: /stackrl/nets
