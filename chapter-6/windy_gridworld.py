import numpy as np
import matplotlib.pyplot as plt

class WindyGridworld():

	def __init__(self, shape, wind_strength):
		assert len(wind_strength) == shape[0], 'Wind strength values are per-column. Length of wind strength array must = shape[0]'

		self.state_shape = np.array(shape, dtype=int)
		self.action_shape = 4

		self.start_pos = np.array([0, shape[1] // 2]) # Start pos at x = 0, y = height/2
		self.goal_pos = np.array([3*shape[0] // 4, shape[1] // 2])

		self.agent_pos = np.zeros(2)
		self.wind_strength = wind_strength

		self.fig = None
	
	def reset(self):
		self.agent_pos = self.start_pos # Set agent to the start
		return self.agent_pos

	def step(self, action):
		wind = self.get_wind()
		move = self.action_to_move(action)
		next_pos = self.agent_pos + move + wind
		
		self.agent_pos = np.clip(next_pos, 0, self.state_shape - 1) # If the agent goes over the bounds, clip him back in
		done = (self.agent_pos == self.goal_pos).all()
		
		return self.agent_pos, -1, done, {}

	def action_to_move(self, action):
		if action == 0:
			move = np.array([0, 1], dtype=int)
		elif action == 1:
			move = np.array([0, -1], dtype=int)
		elif action == 2:
			move = np.array([1, 0], dtype=int)
		elif action == 3:
			move = np.array([-1, 0], dtype=int)

		return move


	def get_wind(self):
		return np.array([0, -self.wind_strength[self.agent_pos[0]]])

	def render(self, delay=1E-5):
		plt.ion()

		img_track = np.zeros(self.state_shape, dtype=int)
		img_track[tuple(self.start_pos)] = 180
		img_track[tuple(self.goal_pos)] = 80
		img_track[tuple(self.agent_pos)] = 128

		if self.fig is None:
			self.fig = plt.figure()
			self.ax = plt.gca()
			
			# Major ticks
			self.ax.set_xticks(np.arange(-0.5, self.state_shape[1] + 0.5, 1))
			self.ax.set_yticks(np.arange(-0.5, self.state_shape[0] + 0.5, 1))

			self.ax.set_xticklabels([])
			self.ax.set_yticklabels([])

			plt.xlim(-0.5, self.state_shape[1] - 0.5)
			plt.ylim(-0.5, self.state_shape[0] - 0.5)
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

		plt.pause(delay)

class WindyGridworldKingMoves(WindyGridworld):

	def __init__(self, shape, wind_strength):
		super().__init__(shape, wind_strength)

		self.action_shape = 8
	
	def action_to_move(self, action):
		if action == 0:
			move = np.array([0, 1], dtype=int)
		elif action == 1:
			move = np.array([0, -1], dtype=int)
		elif action == 2:
			move = np.array([1, 0], dtype=int)
		elif action == 3:
			move = np.array([-1, 0], dtype=int)
		elif action == 4:
			move = np.array([1, 1], dtype=int)
		elif action == 5:
			move = np.array([1, -1], dtype=int)
		elif action == 6:
			move = np.array([-1, 1], dtype=int)
		elif action == 7:
			move = np.array([-1, -1], dtype=int)

		return move


class KingMovesStochastic(WindyGridworldKingMoves):

	def get_wind(self):
		wind = np.array([0, -self.wind_strength[self.agent_pos[0]]])
		p = np.random.random()
		if p < 1/3:
			wind += [0, 1]
		elif p < 2/3:
			wind += [0, -1]

		return wind



		  

