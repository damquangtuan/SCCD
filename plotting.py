import numpy as np
import gym
import gym_gridworld
# /import random
from pylab import *

env = gym.make('GridWorld-v0')

STATE_DIM = env.n_states
ACTION_DIM = env.n_actions



def check():

	actions = [act for act in range(ACTION_DIM)]

	#test to check results
	for idx in range(100):

		obs = env.reset()

		total = 0
		count = 0
		while True:
			# row = int(obs/4) + 1
			# col = int(obs/4) + 1
			#
			# state = [row, col]
			act = random.choice(actions)

			# state = np.reshape(state, [1,STATE_DIM])
			#
			# qvalue_output = sess.run(output, feed_dict={x: state})
			#
			# act = np.argmax(qvalue_output[0])
			#
			next_obs, reward, done, _ = env.step(act)

			env.render()

			total = total + reward
			count = count + 1

			if done:
				break

		print('Final reward:', total)
		print('Count:', count)


	env.close()

def show_cdf(K):

	# Create some test data
	dx = 1
	X = np.arange(0, K, 1)
	text_file = open("total_reward.txt", "r")
	rewards = text_file.read().split(' ')

	rewards = map(float, rewards)
	Y = np.zeros(K)
	for i in range(K):
		Y[i] = rewards[i]

	plot(X, Y, 'ro', marker='o')
	title(r'GridWorld Domain')
	xlabel('K')
	ylabel('Average Rewards')

	show()


show_cdf(459)

