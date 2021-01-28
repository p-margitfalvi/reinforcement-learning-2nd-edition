import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Racetrack():

	@classmethod
	def get_random_track(cls, shape):
		"""
		Create a random right hand turn race track with a given shape. Start line is the bottom edge, end line is the right edge.
		Returns a boolean matrix indicating where the car can go with True values
		shape 	-	array giving map shape as [width, height]
		Returns
		numpy array of bools. Map origin is top left corner, +x to the right, +y down
		"""
		world_map = np.full(shape, False, dtype=bool)

		start_len = np.random.randint(1, shape[0]//2)
		goal_len = np.random.randint(1, shape[1]//2)

		world_map[:start_len, -1] = True # Start line
		world_map[-1, -goal_len:] = True # Start line

		raise NotImplementedError
		return world_map
	
	@classmethod
	def get_right_angle_track(cls, shape, start_len=None, goal_len=None):
		"""
		Create a simple right hand turn race track with a given shape. Start line is the bottom edge, end line is the right edge.
		Returns a boolean matrix indicating where the car can go with True values
		shape 	-	array giving map shape as [width, height]
		Returns
		numpy array of bools. Map origin is top left corner, +x to the right, +y down
		"""

		world_map = np.full(shape, False, dtype=bool)

		start_len = np.random.randint(shape[0]//10, shape[0]//2) if start_len is None else start_len
		goal_len = np.random.randint(shape[1]//10, shape[1]//2) if goal_len is None else goal_len

		world_map[:, :start_len] = True # Start line
		world_map[-goal_len:, :] = True # End line

		return (world_map, (start_len, goal_len))
	
	@classmethod
	def draw_track(cls, track):
		plt.imshow(np.flip(track, axis=0), vmin=0, vmax=1) # TODO: Add colors to start and goal

		ax = plt.gca()

		# Major ticks
		ax.set_xticks(np.arange(-0.5, track.shape[1] + 0.5, 1))
		ax.set_yticks(np.arange(-0.5, track.shape[0] + 0.5, 1))

		ax.set_xticklabels([])
		ax.set_yticklabels([])

		plt.xlim(-0.5, track.shape[1] - 0.5)
		plt.ylim(-0.5, track.shape[0] - 0.5)

		plt.grid()
		plt.show()

	def __init__(self, track=None, shape=None, p_breakdown=0.1):
		assert track or shape, 'Track or shape have to be set'

		if track is None:
			self.track, (self.start_len, self.goal_len) = self.get_right_angle_track(shape=shape)
		else:
			self.track, (self.start_len, self.goal_len) = track
		
		self.fig = None
		self.car_pos = np.array([0, 0])
		self.car_v = np.array([0, 0]) # Car velocity in x and y dir

		self.p_breakdown = p_breakdown
		self.v_max = 5

	def _check_velocity(self, next_vel):
		if next_vel[0] == 0 and next_vel[1] == 0:
			return False
		if min(next_vel) < 0:
			return False
		if max(next_vel) > self.v_max:
			return False
		return True

	def step(self, actions):
		next_pos = self.car_v + self.car_pos

		if not self._check_bounds(next_pos) or not self.track[tuple(next_pos)]:
			# Out of bounds, send back to start
			self.car_pos = np.array([0, np.random.randint(0, self.start_len)], dtype=int)
			self.car_v = np.array([0, 0], dtype=int)
		else:
			next_vel = self.car_v + actions
			if self._check_velocity(next_vel):
				self.car_v = next_vel
			else:
				# Give a random nudge
				r = np.random.random()
				if r < 0.5:
					self.car_v = np.array([0, 1], dtype=int)
				else:
					self.car_v = np.array([1, 0], dtype=int)
			if np.random.random() < self.p_breakdown:
				# Car has broken down, set velocity to 0
				self.car_v = np.array([0, 0], dtype=int)
			self.car_pos = next_pos

		done = False
		rw = -1

		if self.car_pos[1] == self.track.shape[1] - 1:
			# We have reached finish line
			done = True
		
		return np.append(self.car_pos, self.car_v), rw, done, {}
	
	def render(self, delay=1E-5, v=None):
		plt.ion()

		img_track = np.asarray(self.track, dtype=int)*255
		img_track[0, :self.start_len] = 180
		img_track[-self.goal_len:, -1] = 80
		img_track[tuple(self.car_pos)] = 128 # Draw car

		if self.fig is None:
			self.fig = plt.figure()
			self.ax = plt.gca()
			
			# Major ticks
			self.ax.set_xticks(np.arange(-0.5, self.track.shape[1] + 0.5, 1))
			self.ax.set_yticks(np.arange(-0.5, self.track.shape[0] + 0.5, 1))

			self.ax.set_xticklabels([])
			self.ax.set_yticklabels([])

			plt.xlim(-0.5, self.track.shape[1] - 0.5)
			plt.ylim(-0.5, self.track.shape[0] - 0.5)
			plt.grid(True)

			self.img = plt.imshow(np.flip(img_track, axis=0), cmap='gray', vmin=0, vmax=255)
			self.fig.canvas.draw()
			# cache the background
			self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)
			plt.show(block=False)
		
		self.img.set_data(np.flip(img_track, axis=0))
		
		self.fig.canvas.restore_region(self.axbackground)
		self.ax.draw_artist(self.img)
		self.fig.canvas.blit(self.ax.bbox)

		if v is not None:
			for a, z in np.ndenumerate(v):
				self.ax.text(a[1], a[0], '{:0.1f}'.format(z), ha='center', va='center', fontsize=8)

		#self.fig.canvas.flush_events()
		plt.pause(delay)
	
	def _check_bounds(self, next_pos):
		lambda_max = 1
		if not (next_pos[0] >= 0 and next_pos[0] < self.track.shape[0]):
			return False

		if not (next_pos[1] >= 0 and next_pos[1] < self.track.shape[1]):
			return False

		for scale in np.arange(0, lambda_max, 0.1):
			n_pos = np.around(self.car_pos + scale*self.car_v).astype(int)
			if not self.track[tuple(n_pos)]:
				return False

		return True

	def reset(self, p_breakdown=None):
		self.car_pos = np.array([0, np.random.randint(0, self.start_len)], dtype=int)
		self.car_v = np.array([0, 0], dtype=int)
		self.p_breakdown = p_breakdown if p_breakdown is not None else self.p_breakdown

		return np.append(self.car_pos, self.car_v)

class MCOffPolicyAgent():

	def __init__(self, race_env: Racetrack, discount=1):
		track_shape = race_env.track.shape
		self.state_shape = np.append(track_shape, [race_env.v_max + 1, race_env.v_max + 1]) # State is the position of the car and the car velocity in both directions
		self.action_shape = np.array([3, 3]) # Action is velocity change, 3 choices for each direction

		self.env = race_env
		self.discount = discount

		self.Q = np.full(np.append(self.state_shape, self.action_shape), -10) # Corresponding to Q(s, a) where a corresponds to a vector of right actions (-1, 0, 1) and down actions (-1, 0, 1)
		self.C = np.zeros(shape=np.append(self.state_shape, self.action_shape)) # Importance sampling sum
		self.V = np.full(track_shape, -10)
		self.policy = np.zeros(shape=np.append(self.state_shape, 2), dtype=int) # Target policy

	def solve(self, n_episodes=1000):
		behaviour_policy = lambda x: np.array([self.random_action(), self.random_action()])
		behaviour_probs = lambda a, s: 1/9
		
		for i in tqdm(range(n_episodes)):
			trajectory = self.rollout(behaviour_policy, render=(i % 1000==300) and False)
			returns = 0
			weight = 1

			for state, action, reward in reversed(trajectory):
				action_index = action + 1
				returns = self.discount * returns + reward
				self.C[tuple(np.append(state, action_index))] +=  weight
				self.Q[tuple(np.append(state, action_index))] += weight / self.C[tuple(np.append(state, action_index))] * (returns - self.Q[tuple(np.append(state, action_index))])
				self.policy[tuple(state)] = np.array(np.unravel_index(np.argmax(self.Q[tuple(state)]), shape=self.action_shape), dtype=int) - 1
				self.V[state[0], state[1]] = np.max(self.Q[state[0], state[1], :, :])
				if not (action == self.policy[tuple(state)]).all():
					break
				weight *= 1/behaviour_probs(action, state)
			#print(self.Q)
			#print(self.policy)
		
		return self.policy

	def action_to_index(self, action):
		pass

	def random_action(self):
		if np.random.random() < 1/3:
			return -1
		elif np.random.random() < 2/3:
			return 0
		else:
			return 1

	def rollout(self, b_policy, render=False):
		state = self.env.reset()
		done = False
		trajectory = []
		
		while not done:
			if render:
				self.env.render(delay=0.001, v=self.V)
			action = b_policy(state)
			s_prime, rw, done, _ = self.env.step(action)
			#self.env.render(delay=0.001)
			trajectory.append((state, action, rw))
			state = s_prime
		
		return trajectory
	
	def eval_policy(self, n_runs=10):
		average_rw = []
		for i in tqdm(range(n_runs)):
			state = self.env.reset(p_breakdown=0).astype(int)
			done = False
			
			while not done:
				action = self.policy[tuple(state.astype(int))]
				state, rw, done, _ = self.env.step(action)
				average_rw.append(rw)
				self.env.render(delay=0.1, v=self.V)

		return sum(average_rw) / len(average_rw)
	
		


def main():
	shape = [30, 20]
	track = Racetrack.get_right_angle_track(shape=shape, start_len=15, goal_len=20)
	env = Racetrack(track=track)
	print(env.track)
	agent = MCOffPolicyAgent(env)
	trained_policy = agent.solve(n_episodes=10000)
	print(agent.V)
	np.save(f'chapter-5/trained_policy_{shape}', trained_policy)
	print(agent.eval_policy(10))

def load_trained():
	shape = [30, 20]
	track = Racetrack.get_right_angle_track(shape=shape, start_len=15, goal_len=20)

	env = Racetrack(track=track)
	agent = MCOffPolicyAgent(env)
	agent.policy = np.load(f'chapter-5/trained_policy_{shape}.npy')
	print(agent.eval_policy(10))


main()
#load_trained()