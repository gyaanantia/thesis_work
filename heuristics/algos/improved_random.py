from calendar import c
from click import clear
import networkx as nx
import numpy as np
from IPython.display import clear_output
import random
from sdzoo.env.sdzoo import ACTION

from .base import BaseAlgorithm

class ImprovedRandom(BaseAlgorithm):
    ''' Improved random selection, adding more constraints to loading and dropping actions. '''

    def train(self, *args, seed=None, **kwargs):
        ''' Trains the model '''
        pass

    def _reset(self):
        ''' Resets the model '''
        pass

    def evaluate(self, render=False, render_terminal=False, max_cycles=None, max_episodes=1, seed=None):
        ''' Evaluates the model '''

        if max_cycles != None:
            self.env.max_cycles = max_cycles

        for episode in range(max_episodes):
            obs, info = self.env.reset(seed=seed) 
            self._reset()

            if render:
                clear_output(wait=True)
                self.env.render()

            terms = [False]
            truncs = [False]

            self.prevActions = {a: None for a in self.env.agents}

            while not any(terms) and not any (truncs):
                actions = self.generate_actions()
                obs, rewards, terms, truncs, info = self.env.step(actions)

                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

                if render:
                    clear_output(wait=True)
                    self.env.render()

                self.prevActions = actions
            
            if render_terminal:
                clear_output(wait=True)
                self.env.render()

    def generate_actions(self):
        ''' Generates actions for the agents '''
        
        actions = {}
        for agent in self.env.agents:
            if agent.edge is None:
                available_actions = [i for i in range(self.env.sdg.graph.degree(agent.lastNode))]

                if agent.payloads < agent.max_capacity and self.env.sdg.getNodePayloads(agent.lastNode) > 0 and self.env.sdg.getNodePeople(agent.lastNode) == 0:
                    available_actions.append(self.env.action_neighbors_max_degree + ACTION.LOAD)
                if agent.payloads > 0 and self.env.sdg.getNodePeople(agent.lastNode) > 0:
                    available_actions.append(self.env.action_neighbors_max_degree + ACTION.DROP)
            else:
                available_actions = [agent.currentAction]

            actions[agent] = random.choice(available_actions)

        return actions