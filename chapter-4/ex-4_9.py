import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class GamblersProblem():

	def __init__(self, discount=0.99, ph=0.5, goal=100, threshold=0.1):

		self.discount = discount
		self.threshold = threshold

		self.ph = ph # Probability of winning the stake
		self.goal = goal # Terminal state

		self.V = np.zeros(shape=goal + 1)
		self.policy = np.zeros(shape=goal)
		self.V[goal] = 1 # Winning terminal state

	def solve(self):
		self.value_iteration()
		self.policy_selection()

	def value_iteration(self):
		pbar = tqdm()
		max_delta = np.inf

		while max_delta > self.threshold:
			max_delta = 0
			for s in range(1, self.goal):

				old_v = self.V[s]
				self.V[s] = self.compute_value(s)
				delta = np.abs(self.V[s] - old_v)

				if delta > max_delta:
					max_delta = delta
			pbar.update(1)
			pbar.set_postfix({'max_delta': max_delta})

	def policy_selection(self):
		for s in range(1, self.goal):
			self.policy[s] = self.compute_value(s, argmax=True)

	def _transition_probs(self, s_prime, rw, state, action):
		prob = 0
		next_state_win = state + action
		next_state_loss = state - action

		if rw == 1 and (not s_prime == self.goal):
			return 0 # Getting reward of 1 from state other than the winning state is impossible
		
		if s_prime == self.goal and rw == 0:
			return 0 # Getting reward of 0 from the winning state is impossible

		if s_prime == next_state_win:
			return self.ph
		elif s_prime == next_state_loss:
			return 1 - self.ph
		else:
			return 0
		
		print('We shouldnt be getting here')

	def compute_value(self, state, argmax=False): # Computes value of the state for value iteration algorithm
		n_actions = min(state, self.goal - state) + 1
		action_values = np.zeros(n_actions)

		for a in range(n_actions):
			action_values[a] = self.ph * self.V[state + a] + (1 - self.ph) * self.V[state - a] 
		"""for s_prime in range(self.goal + 1):
			for rw in range(2):
				for a in range(n_actions):
					action_values[a] += self._transition_probs(s_prime, rw, state, a) * (rw + self.discount * self.V[s_prime])"""
		if argmax:
			return np.argmax(np.round(action_values[1:], 5)) + 1
		return np.max(action_values)

def main():
	sweep_snapshots = [0, 1, 2, 31]
	solver = GamblersProblem(ph=0.4, discount=1)

	fig, (ax1, ax2) = plt.subplots(1, 2)

	for i in range(40):
		solver.solve()

		if i in sweep_snapshots:
			ax1.step(np.arange(100), solver.V[:-1], label=f'sweep {i + 1}')
	
	print(solver.V)
	print(solver.policy)

	ax1.step(np.arange(100), solver.V[:-1], label='Final estimates')
	
	ax1.grid()
	ax1.legend()
	ax1.set_xlabel('Capital')
	ax1.set_ylabel('Value estimates')

	ax2.step(np.arange(100)[1:], solver.policy[1:])
	ax2.grid()
	ax2.set_xlabel('Capital')
	ax2.set_ylabel('Stake')
	
	plt.show()

def test():
	solver = GamblersProblem(ph=0.4, discount=1)
	
	probs = []
	for state in range(100):
		for a in range(min(state, 100 - state) + 1):
			prob = 0
			for s_prime in range(100 + 1):
				for rw in range(2):
					p = solver._transition_probs(s_prime, rw, 50, 50)
					prob += p
			probs.append(prob)
	print(max(probs))


main()




		
		