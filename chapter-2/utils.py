import numpy as np

class KBandits():

    def __init__(self, k=10, stationary=True):

        self.k = k
        self.stationary = stationary

        if self.stationary:
            self.dist_means = np.random.normal(size=self.k)
        else:
            self.dist_means = np.zeros(self.k)
    
    def step(self, action: int) -> float:
        if not self.stationary:
            self.dist_means += np.random.normal(scale=0.01, size=self.k)
        return np.random.normal(loc=self.dist_means[action])
    
    def reset(self):
        if self.stationary:
            self.dist_means = np.random.normal(size=self.k)
        else:
            self.dist_means = np.zeros(self.k)


class QAgent():

    def __init__(self, alpha, k=10, eps=0.1, initial_Q=None):

        self.k = k
        self.alpha = alpha
        self.eps = eps

        self.Q = np.zeros(k)
        self.N = np.zeros(k)

        self.initial_Q = None
        if initial_Q is not None:
            assert initial_Q.shape == (k,), f'initial_Q argument has to be the same size as k, received size: {initial_Q.shape} instead'
            self.Q = initial_Q
            self.initial_Q = initial_Q

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += self.alpha(self.N[action])*(reward - self.Q[action])
    
    def act(self) -> int:
        r = np.random.random()
        explore = r < self.eps

        if explore:
            action = np.random.randint(self.k)
        else:
            action = np.argmax(self.Q)

        return action

    def reset(self):
        if self.initial_Q is not None:
            self.Q = self.initial_Q
        else:
            self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)

class UCBAgent():

    def __init__(self, k=10, c=1):
        
        self.k = k
        self.step = 1
        self.c = c

        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def update(self, action, reward):
        self.N[action] += 1
        self.step += 1
        self.Q[action] += 1/self.N[action]*(reward - self.Q[action])

    def act(self) -> int:
        return np.argmax(self.Q + self.c*np.sqrt(np.log(self.step)/self.N))
    
    def reset(self):
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)
        self.step = 1

class GradientBandits():

    def __init__(self, k=10, alpha=1):
        
        self.k = k
        self.step = 0
        self.alpha = alpha

        self.reward_mean = 0

        self.H = np.zeros(k)

    def update(self, action, reward):
        logits = self._softmax(self.H)
        self.step += 1
        for a in range(len(self.H)):
            if a == action:
                self.H[action] += self.alpha*(reward - self.reward_mean)*(1 - logits[action])
            else:
                self.H[a] += -self.alpha*(reward - self.reward_mean)*logits[a]
        self.reward_mean += 1/self.step*(reward - self.reward_mean)
                

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def act(self) -> int:
        logits = self._softmax(self.H)
        action = np.random.choice(self.k, p=logits)
        return action
    
    def reset(self):
        self.H = np.zeros(self.k)
        self.step = 0



