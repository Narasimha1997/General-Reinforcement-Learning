import gym
import numpy
from keras.layers import Dense, Flatten
from keras.layers import Activation
from keras.models import load_model, Sequential
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.callbacks import Callback

env = gym.make('CartPole-v1')

def Network():
    model = Sequential()
    model.add(Flatten(input_shape = (1,)+env.observation_space.shape))
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    return model
    pass

#policies:
#callback
class EpsDecayCallback(Callback) :

    def __init__(self, eps_policy, decay_rate = 0.95):
        self.eps_policy = eps_policy
        self.decay_rate = decay_rate
    
    def on_episode_begin(self, episode, logs ={}):
        self.eps_policy.eps*=self.decay_rate

policy = EpsGreedyQPolicy(eps = 1.0)
memory = SequentialMemory(limit = 500000, window_length = 1)

agent = DQNAgent(
    model = Network(), policy = policy, memory = memory, enable_double_dqn = False, nb_actions = env.action_space.n, nb_steps_warmup = 10,
    target_model_update = 1e-2
)

agent.compile(
    optimizer = Adam(lr = 0.002, decay = 2.25e-05), metrics = ['mse']
)

agent.fit(
    env = env, 
    callbacks = [EpsDecayCallback(eps_policy = policy, decay_rate = 0.975)], 
    verbose = 2, 
    nb_steps = 300000
)
agent.save_weights('model.hdf5')

agent.test(env = env, nb_episodes = 100, visualize = True)