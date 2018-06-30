from __future__ import division
import numpy as np
import tensorflow as tf
import random
import gym
import gym_gridworld
import itertools
import os
import pickle
import numpy as np
import time, math
import matplotlib
import matplotlib.pyplot as plt
import byteplay, timeit
# from pylab import *

from collections import defaultdict

env = gym.make("GridWorld-v0")
# env = gym.make("Taxi-v2")


#just for testing
ACTION_DIM = env.n_actions
STATE_DIM = env.n_states

# ACTION_DIM = env.action_space.n
# STATE_DIM  = env.observation_space.n

#parameters
K = 10
max_steps = 100

episodes = 200

LAMBDA = 0.01
GAMMA = 0.9


"""This is sampled trajectory from environment"""
# actions = [0 ,1, 2, 3]
# rewards = [-1, -1, 0, 1, 1]

actions = []
states = []
rewards = []

# reward_epi=[]

#state will be one hot vector for each state
# init_states = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

# states = init_states.astype(np.float32)

def plot_cumulative():

	text_file = open("total_reward.txt", "r")
	reward_epi = text_file.read().split(' ')

	reward_epi = map(float, reward_epi)

	print(reward_epi)
	# plot
	plt.figure(figsize=(15, 10))
	plt.xlabel('Episode', fontsize=20)
	# plt.xlim(-4, episodes+4)
	plt.ylabel('Cumulative Reward', fontsize=20)
	mean_reward = np.mean(reward_epi, axis=0)
	std_reward = np.std(reward_epi, axis=0)
	plt.errorbar(range(K*max_steps), mean_reward, color='b', linewidth=1.5)
	plt.fill_between(range(K*max_steps), (mean_reward - std_reward), (mean_reward + std_reward), color='b', alpha=0.3)
	# plt.savefig('Qlearn_cumulative_reward_curve.png')
	plt.title("SCCD Single Q on Gridworld domain")
	plt.show()

	# fname = 'taxi_Qlearn_s_{}'.format(init_s)
	# save_results(fname, DISCOUNT, LEARNING_RATE, episodes, reward_epi)


def save_data(reward_epi):
	# save reward_epi value matrix to a text file
	mat = np.matrix(reward_epi)
	with open('total_reward.txt', 'wb') as f:
		for line in mat:
			np.savetxt(f, line, fmt='%f')


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

	plot(X, Y, 'ro', marker='.')
	title(r'GridWorld Domain')
	xlabel('K')
	ylabel('Average Rewards')

	show()

def optimise(func):
    c = byteplay.Code.from_code(func.func_code)
    prev=None
    for i, (op, arg) in enumerate(c.code):
        if op == byteplay.BINARY_POWER:
            if c.code[i-1] == (byteplay.LOAD_CONST, 2):
                c.code[i-1] = (byteplay.DUP_TOP, None)
                c.code[i] = (byteplay.BINARY_MULTIPLY, None)
    func.func_code = c.to_code()
    return func

#parameters
def beta(k):
    return k**(-3/4)
	# return 1/k

def nuy(k):
    return k**(-1/2)
	# return 1/(k**2)

def get_state(obs):
	global states
	return states[obs]

def sccd_single_q(sess):
	global states
	# global actions
	# global rewards
	"""This part we construct the QValue Linear Function"""
	# tf Graph Input. This is one hot vector with 1 at dimension k to indicate state k
	x = tf.placeholder(tf.float32, [1,STATE_DIM])
	next_x = tf.placeholder(tf.float32, [1,STATE_DIM])


	# Set model weights
	W = tf.Variable(tf.random_uniform([STATE_DIM, ACTION_DIM], minval=0, maxval=1), dtype=tf.float32)
	b = tf.Variable(tf.random_uniform([1,ACTION_DIM], minval=0, maxval=1), dtype=tf.float32)

	# Variables
	mu = tf.Variable(0, dtype=tf.float32)

	# Construct model. output = xW + b
	# output = tf.matmul(tf.expand_dims(x,0), W) + b
	output = tf.add(tf.matmul(x,W), b)
	# output = tf.matmul(x,W)

	next_output = tf.add(tf.matmul(next_x,W), b)
	# next_output = tf.matmul(next_x,W)


	#format the states for algorithm
	indices = [state for state in range(STATE_DIM)]

	#state considered as features
	states = tf.one_hot(indices, STATE_DIM, dtype=tf.float32)


	init = tf.global_variables_initializer()
	sess.run(init)

	states = sess.run(states)

	result_w = []
	result_b = []


	#format actions for algorithm
	actions = [act for act in range(ACTION_DIM)]

	print(actions)

	reward_epi = np.zeros((K * max_steps))
	env.reset()
	env.render()
	"""This part is the main algorithm"""

	index = 1
	for run in range(K):

		obs = env.reset()

		# each episode

		for k in range(max_steps):

		# while True:
			#randomly chose an action ak and a state sk
			act = random.choice(actions)

			state = get_state(obs)

			state = np.reshape(state, [1,STATE_DIM])

			# print('State:', state)

			# raw_input('wait')

			next_obs, reward, done, _ = env.step(act)

			# total_reward = total_reward + reward

			next_state = get_state(next_obs)

			next_state = np.reshape(next_state, [1,STATE_DIM])


			#derivative of Q at W

			G = LAMBDA* tf.reduce_logsumexp(next_output[0]/LAMBDA) - (output[0][act]/GAMMA)


			d_G_grads = tf.gradients(G, [W, b])
			# d_G_grads = tf.gradients(G, [W])

		# d_G_grads, global_norm = tf.clip_by_global_norm(d_G_grads, 1)

			# [d_G_W, d_G_b] = sess.run(d_G_grads, feed_dict={x:state, next_x: next_state})

			# print('d_G_W', d_G_W)
			# print('d_G_b', d_G_b)


			d_G_W = d_G_grads[0]
			d_G_b = d_G_grads[1]

			# beta_para = beta(k+1+run*max_steps)
			beta_para = beta(index)

			# G_const = sess.run(G, feed_dict={x:state, next_x: next_state})

			#mu
			mu = ((1-beta_para)*mu + beta_para*G)

			#compute d_f at mu
			d_f = 2*GAMMA*(reward + GAMMA*mu)

			# d_f = sess.run(d_f)

			# d_g_W = tf.cast(d_g_W, dtype=tf.float32)

			nuy_para = nuy(index)


			# new_W = W.assign(tf.nn.l2_normalize(W - (d_G_W * (nuy_para * d_f))))

			new_W = W.assign(W - (d_G_W * (nuy_para * d_f)))
			new_b = W.assign(W - (d_G_b * (nuy_para * d_f)))

			# new_b = b.assign(tf.nn.l2_normalize(b - (d_G_b * (nuy_para * d_f))))

			# new_W = W.assign(tf.clip_by_value(W - (d_G_W * (nuy_para * d_f)), 0, np.infty))
			# new_b = b.assign(tf.clip_by_value(b - (d_G_b * (nuy_para * d_f)), 0, np.infty))

			# sess.run([d_q_W_next], feed_dict={x:next_state})
			# [d_f_const, g_k_const, result_mu, result_d_g_W, result_d_q_W_k, result_w, result_b] = sess.run([d_f, g_k, mu, d_g_W, d_q_W_k, new_W, new_b], feed_dict={x:state})

			[result_w, result_b] = sess.run([new_W, new_b], feed_dict={x:state, next_x:next_state})
			# [result_w] = sess.run([new_W], feed_dict={x: state, next_x: next_state})

		# print('result_w: ', result_w)

			# print('result_b', result_b)

			# justtest = raw_input('wait')

			obs = next_obs

			"""To Test"""

			# Run episode

			env_test = gym.make('GridWorld-v0')

			# env_test.seed(10)

			total_reward = 0
			#test 1 times
			for ti in range(1):
				test_obs = env_test.reset()

				for i in range(max_steps):
				# while True:
					state = get_state(test_obs)

					state = np.reshape(state, [1, STATE_DIM])

					qvalue_output = sess.run(output, feed_dict={x: state})

					act = np.argmax(qvalue_output[0])

					testnext_obs, testreward, testdone, _ = env_test.step(act)

					env_test.render()
					test_obs = testnext_obs

					total_reward = total_reward + testreward
					if testdone:
						break

			reward_epi[index-1] = float(total_reward)

			env_test.close()

			""""""""""""""""""""""""""""""""""""
			index = index + 1

			if done:
				break
			# reward_epi[0, k] = qvalue_k

	#after training
	done = True

	rewards_to_return = []

	for i in range(index - 1):
		rewards_to_return.append(reward_epi[i])

	#test to check results
	for idx in range(100):

		total = 0
		obs = env.reset()
		done = False
		for i in range(max_steps):

			state = get_state(obs)

			state = np.reshape(state, [1,STATE_DIM])
			#
			qvalue_output = sess.run(output, feed_dict={x: state})
			#
			act = np.argmax(qvalue_output[0])
			#
			next_obs, reward, done, _ = env.step(act)

			total = total + reward

			# print('reward: ', reward)
			# print('total: ', total)

			if done:
				break

		print('Final reward:', total)

	env.close()

	return [index - 1,rewards_to_return,result_w, result_b]

def sccd_single_v_pi(sess):
	global states
	# global actions
	# global rewards

	"""This part we construct the V Linear Function"""
	# tf Graph Input. This is one hot vector with 1 at dimension k to indicate state k
	x1 = tf.placeholder(tf.float32, [1, STATE_DIM])

	next_x1 = tf.placeholder(tf.float32, [1, STATE_DIM])


	# Set model weights for V
	V_W = tf.Variable(tf.random_uniform([STATE_DIM,1], minval=0, maxval=1), dtype=tf.float32)
	V_b = tf.Variable(tf.random_uniform([1,1], dtype=tf.float32))

	# Construct model. output = xW + b
	V_output = tf.add(tf.matmul(x1, V_W),V_b)
	# V_output = tf.matmul(x1, V_W)


	# V_next_output = tf.add(tf.matmul(next_x1, V_W),V_b)
	V_next_output = tf.matmul(next_x1, V_W)


	"""This part we construct the Policy Linear Function"""
	# Set model weights
	Policy_W = tf.Variable(tf.random_uniform([STATE_DIM, ACTION_DIM], minval=0, maxval=1), dtype=tf.float32)
	Policy_b = tf.Variable(tf.random_uniform([1,ACTION_DIM], minval=0, maxval=1), dtype=tf.float32)

	# Variables
	mu = tf.Variable(0, dtype=tf.float32)

	P_output = tf.add(tf.matmul(x1,Policy_W), Policy_b)
	# P_output = tf.matmul(x1,Policy_W)

	Policy_output = tf.nn.softmax(P_output, axis=1)

	#format the states for algorithm
	indices = [state for state in range(STATE_DIM)]

	#state considered as features
	states = tf.one_hot(indices, STATE_DIM, dtype=tf.float32)


	init = tf.global_variables_initializer()
	sess.run(init)

	states = sess.run(states)

	# To keep the result
	result_w = []
	# result_b = []

	result_Policy_w = []
	# result_Policy_b = []


	#format actions for algorithm
	actions = [act for act in range(ACTION_DIM)]

	reward_epi = np.zeros((K * max_steps))
	env.reset()
	env.render()

	index = 1
	"""This part is the main algorithm"""
	for run in range(K):

		obs = env.reset()
		# each episode
		for k in range(max_steps):

			print('step: ', index)

			#randomly chose an action ak and a state sk
			act = random.choice(actions)

			state = get_state(obs)

			state = np.reshape(state, [1,STATE_DIM])

			next_obs, reward, done, _ = env.step(act)

			# Get the next state

			next_state = get_state(next_obs)

			next_state = np.reshape(next_state, [1,STATE_DIM])

			#G at (sk, ak)
			G = V_next_output[0] - (LAMBDA/GAMMA)*tf.log(Policy_output[0][act]) - (V_output[0]/GAMMA)

			d_grads_V = tf.gradients(G, [V_W, V_b])
			# d_grads_V = tf.gradients(G, [V_W])

			#derivarive of G at V_W and V_b
			d_g_V_W = d_grads_V[0]
			d_g_V_b = d_grads_V[1]

			d_grads_Policy = tf.gradients(G, [Policy_W, Policy_b])
			# d_grads_Policy = tf.gradients(G, [Policy_W])

			d_g_Policy_W = d_grads_Policy[0]
			d_g_Policy_b = d_grads_Policy[1]


			beta_para = beta(index)

			#mu
			mu = (1-beta_para)*mu + beta_para*G

			#compute d_f at mu
			d_f = 2*GAMMA*(reward + GAMMA*mu)

			nuy_para = nuy(index)

			new_V_W = V_W.assign(V_W - d_g_V_W*(nuy_para * d_f))
			new_V_b = V_b.assign(V_b - d_g_V_b*(nuy_para * d_f))


			new_policy_w = Policy_W.assign(Policy_W - d_g_Policy_W * (nuy_para*d_f))
			new_policy_b = Policy_b.assign(Policy_b - d_g_Policy_b * (nuy_para*d_f))

			[result_w, result_b, result_Policy_w, result_Policy_b] = \
				sess.run([new_V_W, new_V_b, new_policy_w, new_policy_b], feed_dict={x1:state, next_x1:next_state})

			# [result_mu, result_d_f, result_w, result_Policy_w] = \
			# 	sess.run([mu, d_f, new_V_W, new_policy_w], feed_dict={x1:state, next_x1:next_state})

			# print('result_mu', result_mu)
			#
			# print('result_d_f', result_d_f)
			#
			# print('result_w', result_w)
			# print('result_b', result_b)
			# print('result_Policy_w', result_Policy_w)
			# print('result_Policy_b', result_Policy_b)


			"""To Test"""
			# Run episode
			env_test = gym.make('GridWorld-v0')
			total_reward = 0
			#test 1 times
			for ti in range(1):
				test_obs = env_test.reset()

				for i in range(max_steps):
				# while True:
					state = get_state(test_obs)

					state = np.reshape(state, [1, STATE_DIM])

					policy_output = sess.run(Policy_output, feed_dict={x1: state})

					act = np.argmax(policy_output[0])

					testnext_obs, testreward, testdone, _ = env_test.step(act)

					env_test.render()
					test_obs = testnext_obs

					total_reward = total_reward + testreward
					if testdone:
						break

			reward_epi[index-1] = float(total_reward)

			env_test.close()

			""""""""""""""""""""""""""""""""""""
			index = index + 1

			if done:
				break

	rewards_to_return = []

	for i in range(index - 1):
		rewards_to_return.append(reward_epi[i])

	return [index - 1 ,rewards_to_return, result_w, result_Policy_w, result_p_w, result_p_b]

def sccd_cyclic(sess):
	global states
	# global actions
	# global rewards

	"""This part we construct the V Linear Function"""
	# tf Graph Input. This is one hot vector with 1 at dimension k to indicate state k
	x1 = tf.placeholder(tf.float32, [1, STATE_DIM])

	next_x1 = tf.placeholder(tf.float32, [1, STATE_DIM])

	# Set model weights for V
	V_W = tf.Variable(tf.random_uniform([STATE_DIM, 1], minval=0, maxval=1), dtype=tf.float32)
	V_b = tf.Variable(tf.random_uniform([1, 1], dtype=tf.float32))

	# Construct model. output = xW + b
	V_output = tf.add(tf.matmul(x1, V_W), V_b)

	V_next_output = tf.add(tf.matmul(next_x1, V_W), V_b)

	"""This part we construct the Policy Linear Function"""
	# Set model weights
	Policy_W = tf.Variable(tf.random_uniform([STATE_DIM, ACTION_DIM], minval=0, maxval=1), dtype=tf.float32)
	Policy_b = tf.Variable(tf.random_uniform([1, ACTION_DIM], minval=0, maxval=1), dtype=tf.float32)

	# Variables
	mu = tf.Variable(0, dtype=tf.float32)

	P_output = tf.add(tf.matmul(x1, Policy_W), Policy_b)

	Policy_output = tf.nn.softmax(P_output, axis=1)

	# format the states for algorithm
	indices = [state for state in range(STATE_DIM)]

	# state considered as features
	states = tf.one_hot(indices, STATE_DIM, dtype=tf.float32)

	init = tf.global_variables_initializer()
	sess.run(init)

	states = sess.run(states)

	# To keep the result
	result_w = []
	result_b = []

	result_Policy_w = []
	result_Policy_b = []

	# format actions for algorithm
	actions = [act for act in range(ACTION_DIM)]

	reward_epi = np.zeros((K * max_steps))
	env.reset()
	env.render()

	index = 1
	"""This part is the main algorithm"""
	for run in range(K):

		obs = env.reset()
		# each episode
		for k in range(max_steps):

			print('step: ', index)

			# randomly chose an action ak and a state sk
			act = random.choice(actions)

			state = get_state(obs)

			state = np.reshape(state, [1, STATE_DIM])

			next_obs, reward, done, _ = env.step(act)

			# Get the next state

			next_state = get_state(next_obs)

			next_state = np.reshape(next_state, [1, STATE_DIM])

			# G at (sk, ak)
			G = V_next_output[0] - (V_output[0] / GAMMA)

			d_grads_V = tf.gradients(G, [V_W, V_b])

			# derivarive of G at V_W and V_b
			d_g_V_W = d_grads_V[0]
			d_g_V_b = d_grads_V[1]


			beta_para = beta(index)

			# mu
			mu = (1 - beta_para) * mu + beta_para * G

			tmp_log = tf.log(Policy_output[0][act])

			tmp_log_grads = tf.gradients(tmp_log, [Policy_W, Policy_b])

			# compute d_f at mu
			d_f_V = 2 * GAMMA * (reward + GAMMA * mu - LAMBDA * tf.log(Policy_output[0][act]))
			d_f_P_W = -2 * LAMBDA * (reward + GAMMA * mu - LAMBDA * tf.log(Policy_output[0][act])) * tmp_log_grads[0]
			d_f_P_b = -2 * LAMBDA * (reward + GAMMA * mu - LAMBDA * tf.log(Policy_output[0][act])) * tmp_log_grads[1]


			nuy_para = nuy(index)

			new_V_W = V_W.assign(V_W - d_g_V_W * (nuy_para * d_f_V))
			new_V_b = V_b.assign(V_b - d_g_V_b * (nuy_para * d_f_V))

			new_policy_w = Policy_W.assign(Policy_W - (nuy_para * d_f_P_W))
			new_policy_b = Policy_b.assign(Policy_b - (nuy_para * d_f_P_b))

			[result_w, result_b, result_Policy_w, result_Policy_b] = \
				sess.run([new_V_W, new_V_b, new_policy_w, new_policy_b],
						 feed_dict={x1: state, next_x1: next_state})

			# print('result_mu', result_mu)
			#
			# print('result_d_f', result_d_f)
			#
			# print('result_w', result_w)
			# print('result_b', result_b)
			# print('result_Policy_w', result_Policy_w)
			# print('result_Policy_b', result_Policy_b)

			"""To Test"""
			# Run episode
			env_test = gym.make('GridWorld-v0')
			total_reward = 0
			# test 1 times
			for ti in range(1):
				test_obs = env_test.reset()

				for i in range(max_steps):
					# while True:
					state = get_state(test_obs)

					state = np.reshape(state, [1, STATE_DIM])

					policy_output = sess.run(Policy_output, feed_dict={x1: state})

					act = np.argmax(policy_output[0])

					testnext_obs, testreward, testdone, _ = env_test.step(act)

					env_test.render()
					test_obs = testnext_obs

					total_reward = total_reward + testreward
					if testdone:
						break

			reward_epi[index - 1] = float(total_reward)

			env_test.close()

			""""""""""""""""""""""""""""""""""""
			index = index + 1

			if done:
				break

	rewards_to_return = []

	for i in range(index - 1):
		rewards_to_return.append(reward_epi[i])

	return [index - 1, rewards_to_return, result_w, result_b, result_Policy_w, result_Policy_b]


if __name__=='__main__':
	#beta = optimise(beta)
	#nuy = optimise(nuy)

	with tf.Session() as sess:

		# [K, reward_epi, result_v_w, result_p_w] = sccd_single_v_pi(sess)
		[K,reward_epi,result_w, result_b] = sccd_single_q(sess)
		#[K, reward_epi, result_w, result_b, result_p_w, result_p_b] = sccd_cyclic(sess)


		# state = states[2]

		# print('state: ', state)
		print('result_v_w: ', result_w)
		print('result_p_w ', result_b)
		# print('P_W: ', result_p_w)
		# print('P_b ', result_p_b)

	# reward_epi = np.zeros(20)
	print(K)
	save_data(reward_epi)
	# show_cdf(K)
	# print('xW + b:', np.add(np.matmul(np.transpose(state), result_w), result_b))
