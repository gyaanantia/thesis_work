from IPython.display import clear_output
from sdzoo.sdzoo_v0 import SDGraph, parallel_env
from sdzoo.env.communication_model import CommunicationModel
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt 
import pandas as pd

from algos.random import RandomChoice
from algos.improved_random import ImprovedRandom
from algos.force_load_drop import ForceLoadDrop


MAX_CYCLES = 200
MAX_EPISODES = 2500
RENDER_ALL = False
RENDER_TERMINAL = False

sdg = SDGraph("../sdzoo/env/9nodes.graph")
env = parallel_env(sdg, 4,
                    speed = 40, 
                    observe_method="pyg", 
                    alpha=1.0,
                    beta=10.0,
                    load_reward=1.0,
                    drop_reward=1.0,
                    state_reward=1.0,
                    step_penalty=0.05,
                    step_reward=1.0)

algo1 = RandomChoice(env)
algo2 = ImprovedRandom(env)
algo3 = ForceLoadDrop(env)

rewards1 = algo1.evaluate(1, render=RENDER_ALL, render_terminal=RENDER_TERMINAL, max_cycles=MAX_CYCLES, max_episodes=MAX_EPISODES)
rewards2 = algo2.evaluate(2, render=RENDER_ALL, render_terminal=RENDER_TERMINAL, max_cycles=MAX_CYCLES, max_episodes=MAX_EPISODES)
rewards3 = algo3.evaluate(3, render=RENDER_ALL, render_terminal=RENDER_TERMINAL, max_cycles=MAX_CYCLES, max_episodes=MAX_EPISODES)

rewards = pd.concat([rewards1, rewards2['reward2'], rewards3['reward3']], axis=1)
rewards.to_csv("heuristic_rewards.csv")

print("DONE")