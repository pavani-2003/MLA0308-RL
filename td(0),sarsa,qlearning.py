import numpy as np

# Define the environment
class Environment:
    def __init__(self):
        self.num_states = 10
        self.num_actions = 2
        self.transition_matrix = np.random.rand(self.num_states, self.num_actions, self.num_states)
        self.rewards = np.random.randint(0, 10, size=(self.num_states, self.num_actions))

    def step(self, state, action):
        next_state = np.random.choice(range(self.num_states), p=self.transition_matrix[state, action])
        reward = self.rewards[state, action]
        return next_state, reward

# Define Train A with TD(0)
class TrainA:
    def __init__(self, environment, num_steps):
        self.environment = environment
        self.num_steps = num_steps
        self.values = np.zeros(environment.num_states)

    def train(self):
        state = 0
        for _ in range(self.num_steps):
            action = np.random.randint(0, self.environment.num_actions)
            next_state, reward = self.environment.step(state, action)
            self.values[state] += reward
            state = next_state

# Define Train B with SARSA
class TrainB:
    def __init__(self, environment, num_steps):
        self.environment = environment
        self.num_steps = num_steps
        self.values = np.zeros(environment.num_states)

    def train(self):
        state = 0
        for _ in range(self.num_steps):
            action = np.random.randint(0, self.environment.num_actions)
            next_state, reward = self.environment.step(state, action)
            self.values[state] += reward
            next_action = np.random.randint(0, self.environment.num_actions)
            state = next_state

# Define Train C with Q-Learning
class TrainC:
    def __init__(self, environment, num_steps):
        self.environment = environment
        self.num_steps = num_steps
        self.q_values = np.zeros((environment.num_states, environment.num_actions))

    def train(self):
        state = 0
        for _ in range(self.num_steps):
            action = np.argmax(self.q_values[state])
            next_state, reward = self.environment.step(state, action)
            self.q_values[state, action] += reward
            state = next_state

# Simulation
env = Environment()
num_steps = 1000

train_a = TrainA(env, num_steps)
train_b = TrainB(env, num_steps)
train_c = TrainC(env, num_steps)

train_a.train()
train_b.train()
train_c.train()

# Compare results
reward_a = sum(train_a.values)
reward_b = sum(train_b.values)
reward_c = sum(train_c.q_values.max(axis=1))

print("Total rewards:")
print("Train A (TD(0)): ", reward_a)
print("Train B (SARSA): ", reward_b)
print("Train C (Q-Learning): ", reward_c)
