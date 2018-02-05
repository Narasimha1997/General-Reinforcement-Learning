import gym
from keras.models import load_model
from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential
import numpy

env = gym.make('MountainCar-v0')

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
    model.load_weights('model.hdf5')
    return model

model = Network()


print(env.observation_space)

while True:
    rew = 0; steps = 0
    o = env.reset()
    while True:
        env.render()
        action = numpy.argmax(
            model.predict(numpy.reshape(o, (1, 1,)+env.observation_space.shape))
        )
        o , r, d, i = env.step(action)
        rew+=r; steps+=1
        if d:
            print('Average episode reward: ', (rew/steps))
            break