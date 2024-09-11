import numpy as np
import torch
from RL_brain import DQN,ReplayBuffer

def run_this(lr, capacity):
    n_states = 4  # Example feature space size
    n_actions = 2  # Example action space size
    n_hidden = 50  # Example hidden layer size

    # Initialize DQN with given learning rate and memory size
    dqn = DQN(n_states, n_hidden, n_actions, learning_rate=lr, gamma=0.99, epsilon=0.1, target_update=10, device=torch.device("cpu"))
    replay_buffer = ReplayBuffer(capacity)

    episodes = 10  # Example number of episodes
    return_list = []

    for episode in range(episodes):
        total_reward = 0
        observation = np.random.rand(n_states)  # Example initial observation
        while True:
            action = dqn.take_action(observation)
            observation_, reward, done = np.random.rand(n_states), np.random.rand(), np.random.choice([True, False])  # Example environment step
            replay_buffer.add(observation, action, reward, observation_, done)
            total_reward += reward
            if replay_buffer.size() > 32:
                batch = replay_buffer.sample(32)
                transition_dict = {
                    'states': batch[0],
                    'actions': batch[1],
                    'rewards': batch[2],
                    'next_states': batch[3],
                    'dones': batch[4]
                }
                dqn.update(transition_dict)
            if done:
                break
            observation = observation_
        return_list.append(total_reward)
    return np.mean(return_list)

if __name__ == "__main__":
    from GA import GeneticAlgorithm

    pop_size = 20
    gene_length = 2
    crossover_rate = 0.7
    mutation_rate = 0.1
    generations = 10

    ga = GeneticAlgorithm(pop_size, gene_length, crossover_rate, mutation_rate, generations)
    best_params = ga.optimize()
    print("Best parameters found: Learning Rate = {}, Memory Size = {}".format(best_params[0], int(best_params[1])))
