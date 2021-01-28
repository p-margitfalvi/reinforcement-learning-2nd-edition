import numpy as np
import scipy
from tqdm import tqdm

class CarRental():

	def __init__(self, n_locations):
		self.n_locations = n_locations
		self.cars = np.zeros(self.n_locations)
		self.means_requests = [3, 4]
		self.means_returns = [3, 2]
		self.max_cars = 20

	def reset(self):
		self.cars = np.zeros(self.n_locations)
		self._return_cars()

	def step(self, action):
		cost = self._move_cars(action)
		profit = self._request_cars()
		self._return_cars()

		return profit - cost

	def _return_cars(self):
		for i in range(self.n_locations):
			self.cars[i] += np.random.poisson(lam=self.means_returns[i])
			if self.cars[i] > self.max_cars:
				self.cars[i] = self.max_cars

	def _request_cars(self):
		profit = 0
		for i in range(self.n_locations):
			request = np.random.poisson(lam=self.means_requests[i])
			clipped_request = min(request, self.cars[i]) # Clip to never take more cars than available
			profit += clipped_request * 10 # 10$ for each car rented
			self.cars[i] -= clipped_request
		return profit
	
	def _move_cars(self, action):
		cost = 0
		action = min(max(action, -5), 5) # Clip action within range [-5, 5]
		if action > 0:
			# Move from 1 to 2
			clipped_action = min(action, self.cars[0]) # Never take more cars than available
			self.cars[0] -= clipped_action
			self.cars[1] = min(clipped_action + self.cars[1], self.max_cars) # Ensure there's not more than max cars at the other location
		else:
			# Move from 2 to 1
			clipped_action = min(action, self.cars[1]) # Never take more cars than available
			self.cars[1] -= clipped_action
			self.cars[0] = min(clipped_action + self.cars[1], self.max_cars) # Ensure there's not more than max cars at the other location
		cost = clipped_action * 2
		return cost

class DPSolver():

	def __init__(self, discount, threshold=0.1):
		self.V = np.zeros(shape=(21, 21)) # Initiliase arbitrarily
		self.policy = np.zeros(shape=(21, 21))
		self.prev_policy = np.zeros(shape=(21, 21))

		tmp_x, tmp_y = np.meshgrid(np.arange(21), np.arange(21))
		tmp_x = tmp_x.flatten()
		tmp_y = tmp_y.flatten()

		self.state_space = np.vstack((tmp_x, tmp_y)).T

		self.threshold = threshold
		self.discount = discount

		self.poisson_cache = dict()

	
	def solve(self):
		policy_stable = False
		pbar = tqdm()

		while not policy_stable:
			self._policy_evaluation()
			policy_stable = self._policy_improvement()
			pbar.update(1)
		
		np.save('value_car_rental.npy', self.V)
		np.save('policy_car_rental.npy', self.policy)

	def _policy_improvement(self): # Improve policy by acting greedily w.r. to the value function
		policy_stable = True
		for loc1 in range(21):
			for loc2 in range(21):
				state = np.array([loc1, loc2])
				old_action = self.policy[loc1, loc2]
				self.policy[loc1, loc2] = self._compute_value(state)
				if not (old_action == self.policy[loc1, loc2]):
					policy_stable = False
		return policy_stable


	def _policy_evaluation(self): # Update values of the states
		max_delta = np.inf
		pbar = tqdm()

		while max_delta > self.threshold:
			pbar.update(1)
			max_delta = 0
			for loc1 in range(21): # Iterate over all the states
				for loc2 in range(21):
					
					state = np.array([loc1, loc2])
					action = self.policy[loc1, loc2]
					old_val = self.V[loc1, loc2]
					
					self.V[loc1, loc2] = self._compute_value(state, action)
					
					if np.abs(self.V[loc1, loc2] - old_val) > max_delta:
						max_delta = np.abs(self.V[loc1, loc2] - old_val)
	
	def _compute_value(self, state, action=None):
		value = 0
		
		if action is None:
			value = np.zeros(5*2+1)

		for loc1 in range(21):
			for loc2 in range(21):
				for n_rented in range(loc1 + loc2 + 1):
					rw = n_rented * 10
					s_prime = np.array([loc1, loc2])
					if action is None:
						for idx, a in enumerate(range(max(-5, loc2), min(6, loc1 + 1))):
							value[idx] += self.transition_probs(s_prime, rw, state, a) * (rw + self.discount * self.V[loc1, loc2])
					else:
						value += self.transition_probs(s_prime, rw, state, self.policy[loc1, loc2]) * (rw + self.discount * self.V[loc1, loc2])
		
		if action is None:
			return np.argmax(value)

		return value

	def transition_probs(self, s_prime, rw, s, action):
		#probabilities = np.zeros(shape=(21, 21))
		state = s
		if action > 0: # Move from 1 to 2
			if (state[0] < action): # Not enough cars, not possible
				return 0
			state[0] -= action
			state[1] = min(state[1] + action, 20)
			cost = 2 * action
		else:
			if (state[1] < action):
				return 0
			state[1] -= action
			state[0] = min(state[0] + action, 20)
			cost = -2 * action

		cars_needed = s_prime - state
		prob = 1
		total_requested = (rw + cost) // 10
		prob = 0
		
		for requested_1 in range(int(total_requested)):
			requested_2 = total_requested - requested_1

			returned_1 = cars_needed[0] + requested_1
			returned_2 = cars_needed[1] + requested_2

			if (min(returned_1, returned_2) < 0 or max(returned_1, returned_2) >= 20):
				continue

			prob_1 = self._p_poisson(returned_1, lam=3.0) * self._p_poisson(requested_1, lam=3.0) # Probability that we get the right numbers to satisfy the reward and s_prime
			prob_2 = self._p_poisson(returned_2, lam=2.0) * self._p_poisson(requested_2, lam=4.0)

			prob += prob_1 * prob_2

		return prob


	def _p_poisson(self, n, lam=1.0):
		assert n >= 0, 'Sample has to be larger than or equal to 0'
		key = n * 10 + lam
		if not key in self.poisson_cache:
			self.poisson_cache[key] = lam**n/np.math.factorial(n)*np.exp(-lam)
		return self.poisson_cache[key]


def main():
	solver = DPSolver(discount=0.99)
	solver.solve()


main()

