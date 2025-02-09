# Import necessary libraries
import chess
import chess.svg
import random

# Initialize the chess board
board = chess.Board()

# Define a simple random move agent
def random_move_agent(board):
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)

# Play a game
while not board.is_game_over():
    print(board)
    move = random_move_agent(board)
    board.push(move)

print("Game over. Result:", board.result())



pip install gym numpy

import gym
from gym import spaces
import numpy as np

class SimplifiedChessEnv(gym.Env):
    def __init__(self):
        super(SimplifiedChessEnv, self).__init__()
        self.board_size = 4
        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int)
        self.current_pos = (0, 0)
        self.board[self.current_pos] = 1
        return self.board

    def step(self, action):
        x, y = divmod(action, self.board_size)
        if self.board[x, y] == 1:
            reward = -1  # Invalid move (already occupied)
            done = True
        else:
            self.board[self.current_pos] = 0
            self.current_pos = (x, y)
            self.board[self.current_pos] = 1
            reward = 1
            done = False
        return self.board, reward, done, {}

    def render(self, mode='human'):
        for row in self.board:
            print(' '.join(str(cell) for cell in row))
        print()

import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values

    def update(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

        if done:
            self.epsilon *= self.epsilon_decay

def state_to_tuple(state):
    return tuple(map(tuple, state))
`

env = SimplifiedChessEnv()
agent = QLearningAgent(env.action_space)

n_episodes = 1000
max_steps_per_episode = 100

for episode in range(n_episodes):
    state = env.reset()
    state = state_to_tuple(state)
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = state_to_tuple(next_state)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

print("Training finished.")


state = env.reset()
state = state_to_tuple(state)
env.render()

total_reward = 0
done = False

while not done:
    action = agent.select_action(state)
    state, reward, done, _ = env.step(action)
    state = state_to_tuple(state)
    total_reward += reward
    env.render()

print(f"Total Reward: {total_reward}")
