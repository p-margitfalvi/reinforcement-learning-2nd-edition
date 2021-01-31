import numpy as np
from tqdm import tqdm
from windy_gridworld import *

class SARSA():

	def __init__(self, env, alpha=0.1, discount=1, eps=0.1, Q_init=0):
		
		self.state_shape = env.state_shape
		self.action_shape = env.action_shape
		self.state_action_shape = np.append(self.state_shape, self.action_shape)

		self.env = env
		
		self.discount = discount
		self.alpha = alpha
		self.eps = eps

		self.Q = np.full(self.state_action_shape, Q_init, dtype=float)
		self.Q[env.goal_pos] = 0 # Set terminal state to zero
		self.policy = np.random.choice(4, self.state_shape)

	def train(self, episodes=100):
		for ep in tqdm(range(episodes)):
			done = False
			state = self.env.reset()
			action = self.policy[tuple(state)]
			while not done:
				if np.random.random() < self.eps: # Take random action
					action = np.random.choice(4, size=1) # Sample for both actions

				#self.env.render()
				s_prime, rw, done, _ = self.env.step(action)
				
				a_prime = self.policy[tuple(s_prime)] # Compute next action from current policy

				# Update action value function
				self.Q[tuple(np.append(state, action))] += self.alpha * (rw + self.discount * self.Q[tuple(np.append(s_prime, a_prime))] - self.Q[tuple(np.append(state, action))])
				# Update policy by being greedy wrt to Q
				self.policy[tuple(state)] = np.argmax(self.Q[tuple(state)])
				
				state = s_prime
				action = a_prime

	def action_to_index(self, action): # Maps action -1 to index 0 and action 1 to index 1 
		return (action + 1) / 2

	def index_to_action(self, index): # Maps index 0 to action -1 and index 1 to 1
		return 2 * index - 1

	def policy_evaluate(self, episodes=10):
		av_steps = []
		for ep in tqdm(range(episodes)):
			done = False
			state = self.env.reset()
			step = 0
			while not done:
				action = self.policy[tuple(state)]
				self.env.render()
				state, rw, done, _ = self.env.step(action)
				step += 1
			av_steps.append(step)

		return sum(av_steps)/len(av_steps)


def main():
	wind_strengths = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
	env = WindyGridworld([10, 7], wind_strengths)
	agent = SARSA(env, alpha=0.5, Q_init=0)
	agent.train(200)
	print(agent.policy_evaluate(episodes=10))

	env_king = WindyGridworldKingMoves([10, 7], wind_strengths)
	agent = SARSA(env_king, alpha=0.5, Q_init=0)
	agent.train(500)
	print(agent.policy_evaluate(episodes=10))

	env_king = KingMovesStochastic([10, 7], wind_strengths)
	agent = SARSA(env_king, alpha=0.5, Q_init=0)
	agent.train(500)
	print(agent.policy_evaluate(episodes=10))

main()