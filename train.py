from spinup import ppo
from spinup import ddpg
import tensorflow as tf
import gym
import gym_foo


env_fn = lambda : gym.make('foo-v2')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='./output', exp_name='experiment_name')

#ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=5, logger_kwargs=logger_kwargs)

ddpg(env_fn=env_fn, ac_kwargs=ac_kwargs, seed=0,
         steps_per_epoch=5000, epochs=50, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
         act_noise=0.1, max_ep_len=1000, logger_kwargs=logger_kwargs, save_freq=1)
