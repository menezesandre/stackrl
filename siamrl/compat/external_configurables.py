from tf_agents.metrics import tf_metrics
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from gin import external_configurable as ec

# Metrics
ec(tf_metrics.AverageReturnMetric, module='tf_agents.metrics')
ec(tf_metrics.ChosenActionHistogram, module='tf_agents.metrics')
ec(tf_metrics.EnvironmentSteps, module='tf_agents.metrics')
ec(tf_metrics.NumberOfEpisodes, module='tf_agents.metrics')

# Agents
ec(dqn_agent.DqnAgent, module='tf_agents.agents')
ec(dqn_agent.DdqnAgent, module='tf_agents.agents')

# Replay buffers
ec(tf_uniform_replay_buffer.TFUniformReplayBuffer, module='tf_agents.replay_buffers')