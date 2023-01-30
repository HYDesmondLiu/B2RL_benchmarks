#!/usr/bin/env python3.6

import gym
import sinergym
from sinergym.utils.wrappers import NormalizeObservation
from sinergym.utils.constants import RANGES_5ZONE

import numpy as np
import torch
import gym
import argparse
import os

import utils
#import TD3_BC

import json
import d3rlpy

# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, seed_offset=100, eval_episodes=1):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			#action = policy.select_action(state)
			action = policy.predict(state.reshape(1,-1))
			state, reward, done, _ = eval_env.step(action.reshape(-1,1))
			avg_reward += reward

	avg_reward /= eval_episodes

	with open('/data/hsinyu/01_Building/sinergym/Eplus-5Zone_buffers.json', 'r') as fp:
			Eplus5Zone_dict = json.load(fp)
			env_expert, env_random = Eplus5Zone_dict[env_name][2], Eplus5Zone_dict[env_name][1]
			normalized_score = (avg_reward - env_random)/(env_expert - env_random)

	print("---------------------------------------")
	print(f"Raw score: {avg_reward}")
	print(f"Evaluation over {eval_episodes} episodes: {normalized_score:.3f}")
	print("---------------------------------------")
	return normalized_score


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="CQL")               # Policy name
	parser.add_argument("--env", default="Eplus-5Zone-mixed-continuous-stochastic-v1")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=25e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=5e5, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# TD3 + BC
	parser.add_argument("--alpha", default=2.5)
	parser.add_argument("--normalize", default=True)
	parser.add_argument("--buffer_folder", default='/data/hsinyu/01_Building/sinergym/01_DDPG/buffers/')
	# CQL
	parser.add_argument("--n_steps", default=5000, type=int) 
	
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)
	for key,value in RANGES_5ZONE.items():
		print(f'{key} {value}')

	env = NormalizeObservation(env,ranges=RANGES_5ZONE)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	print(f'State dim.:{state_dim} / Action dim.:{action_dim} / Max action:{max_action}')
	
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		# TD3
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		# TD3 + BC
		"alpha": args.alpha
	}

	# Initialize policy
	if args.policy == 'CQL':
		#policy = TD3_BC.TD3_BC(**kwargs)
		policy = d3rlpy.algos.CQL(
			use_gpu = True,
		)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.load(f'{args.buffer_folder}/{args.env}')
	
	reward = replay_buffer.reward
	action = replay_buffer.action
	state = replay_buffer.state
	#next_state = replay_buffer.next_state
	not_done = replay_buffer.not_done

	dataset = d3rlpy.dataset.MDPDataset(
		observations = state,
		actions = action,
		rewards = reward,
		terminals = not_done,
	)

	evaluations = []
	for t in range(int(args.max_timesteps/args.eval_freq)):

		#policy.train(replay_buffer, args.batch_size)
		policy.fit(dataset, 
			n_steps = int(args.eval_freq), 
			n_steps_per_epoch = int(args.eval_freq),
			)
		# Evaluate episode
		#if (t + 1) % args.eval_freq == 0:
		print(f"Time steps: {t+1}")
		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/{file_name}", evaluations)
		if args.save_model: policy.save(f"./models/{file_name}")
