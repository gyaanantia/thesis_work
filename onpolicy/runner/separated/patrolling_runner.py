from collections import defaultdict, deque
from itertools import chain
import os
import time
import copy

import imageio
import numpy as np
import torch
import wandb

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()

class PatrollingRunner(Runner):
    def __init__(self, config):

        # The default restore functionality is broken. Disable it and do it ourselves.
        # We make a copy of the config so we can modify it without affecting the original.
        configNoRestore = copy.deepcopy(config)
        configNoRestore['all_args'].model_dir = None

        super(PatrollingRunner, self).__init__(configNoRestore)
        self.env_infos = defaultdict(list)

        if self.use_centralized_V:
            # Set up a shared critic. This is slightly hacky!
            # We abuse the existing classses and just use the critic from the first agent.
            self.critic = self.policy[0].critic
            self.critic_optimizer = self.policy[0].critic_optimizer
            for po in self.policy:
                po.critic = self.critic
                po.critic_optimizer = self.critic_optimizer
            for ta in self.trainer:
                if ta._use_popart:
                    ta.value_normalizer = self.critic.v_out
        
        # Set up additional replay buffers for asynchronous actors.
        if self.all_args.skip_steps_async:
            self.buffer = [[] for i in range(self.n_rollout_threads)]
            for i in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]

                    # make a copy of all_args but with n_rollout_threads = 1, since we are hacking this to use a separate buffer per rollout thread per agent.
                    args = copy.deepcopy(self.all_args)
                    args.n_rollout_threads = 1
                    bu = SeparatedReplayBuffer(args,
                                            self.envs.observation_space[agent_id],
                                            share_observation_space,
                                            self.envs.action_space[agent_id])
                    self.buffer[i].append(bu)
        
        # Perform restoration.
        self.model_dir = config['all_args'].model_dir
        if self.model_dir is not None:
            self.restore()
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # Create a matrix indicating whether each agent in each environment is ready for a new action.
            self.ready = np.zeros((self.n_rollout_threads, self.num_agents), dtype=bool)
            for i in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    self.ready[i, a] = True

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                combined_obs, rewards, dones, infos = self.envs.step(actions_env)

                # Pull agent ready state from the info message.
                for i in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        self.ready[i, a] = infos[i]["ready"][a]

                # Split the combined observations into obs and share_obs, then combine across environments.
                obs = []
                share_obs = []
                for o in combined_obs:
                    obs.append(o["obs"])
                    share_obs.append(o["share_obs"][0])
                obs = np.array(obs)
                share_obs = np.array(share_obs)

                # Get the delta steps from the environment info.
                delta_steps = np.array([info["deltaSteps"] for info in infos])
                # delta_steps = np.array(delta_steps).reshape(-1, 1)

                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                
                avgEpRewards = np.mean([self.buffer[i][a].rewards for a in range(self.num_agents) for j in range(self.n_rollout_threads)]) * self.episode_length
                if self.use_wandb:
                    wandb.log({"average_episode_rewards": avgEpRewards}, step=total_num_steps)
                else:
                    self.writter.add_scalars("average_episode_rewards", {"average_episode_rewards": avgEpRewards}, total_num_steps)

                print("average episode rewards is {}".format(avgEpRewards))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        combined_obs = self.envs.reset()

        # Split the combined observations into obs and share_obs, then combine across environments.
        obs = []
        share_obs = []
        for o in combined_obs:
            obs.append(o["obs"])
            share_obs.append(o["share_obs"][0])
        obs = np.array(obs)
        share_obs = np.array(share_obs)

        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if not self.use_centralized_V:
                    share_obs = np.array(list(obs[i, agent_id]))

                self.buffer[i][agent_id].share_obs[0] = share_obs[i].copy()
                self.buffer[i][agent_id].obs[0] = np.array(list(obs[i, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        actions = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        action_log_probs = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        rnn_states = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        rnn_states_critic = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        actions_env = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]

        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                value, action, action_log_prob, rnn_state, rnn_state_critic \
                    = self.trainer[agent_id].policy.get_actions(self.buffer[i][agent_id].share_obs[step],
                                                                self.buffer[i][agent_id].obs[step],
                                                                self.buffer[i][agent_id].rnn_states[step],
                                                                self.buffer[i][agent_id].rnn_states_critic[step],
                                                                self.buffer[i][agent_id].masks[step])
                # [agents, envs, dim]
                values[i][agent_id] = _t2n(value)[0]
                action = _t2n(action)[0]

                actions[i][agent_id] = action
                action_log_probs[i][agent_id] = _t2n(action_log_prob)[0]
                rnn_states[i][agent_id] = _t2n(rnn_state)[0]
                rnn_states_critic[i][agent_id] = _t2n(rnn_state_critic)[0]

                actions_env[i][agent_id] = action[0]
        
        actions_env = np.array(actions_env)

        values = np.array(values)#.transpose(1, 0, 2)
        actions = np.array(actions)#.transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs)#.transpose(1, 0, 2)
        rnn_states = np.array(rnn_states)#.transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic)#.transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps = data
        
        # Add the average idleness time to env infos.
        self.env_infos["avg_idleness"] = [i["avg_idleness"] for i in infos]

        # reset rnn and mask args for done envs
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)


        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if self.use_centralized_V:
                    s_obs = share_obs[i]
                else:
                    s_obs = np.array(list(obs[i, agent_id]))

                self.buffer[i][agent_id].insert(s_obs,
                                            np.array(list(obs[i, agent_id])),
                                            rnn_states[i, agent_id],
                                            rnn_states_critic[i, agent_id],
                                            actions[i, agent_id],
                                            action_log_probs[i, agent_id],
                                            values[i, agent_id],
                                            rewards[i, agent_id],
                                            masks[i, agent_id],
                                            delta_steps[i, agent_id])

    def log_train(self, train_infos, total_num_steps): 
        # The train_infos is a list (size self.n_rollout_threads) of lists (size self.num_agents) of dicts.
        # We want to flatten this to a single list of dicts by averaging across rollout threads.
        train_infos = [dict(chain.from_iterable(d.items() for d in agent_infos)) for agent_infos in zip(*train_infos)]

        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self, ipython_clear_output=True):        

        if ipython_clear_output:
            from IPython.display import clear_output

        # reset envs and init rnn and mask
        render_env = self.envs

        for i_episode in range(self.all_args.render_episodes):
            combined_obs = render_env.reset()
            render_actions = np.zeros((self.n_render_rollout_threads, self.num_agents), dtype=np.float32)
            render_rnn_states = np.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)

            # Split the combined observations into obs and share_obs, then combine across environments.
            obs = []
            share_obs = []
            for o in combined_obs:
                obs.append(o["obs"])
                share_obs.append(o["share_obs"][0])
            obs = np.array(obs)
            share_obs = np.array(share_obs)

            # Record the readiness of each agent for a new action. All agents ready by default.
            ready = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=bool)


            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                # We only use a single rollout thread for rendering.
                for i in range(self.n_render_rollout_threads):
                    for agent_id in range(self.num_agents):
                        if not ready[i, agent_id]:
                            continue

                        if agent_id == 0:
                            print('hello')

                        self.trainer[agent_id].prep_rollout()
                        render_action, render_rnn_state = self.trainer[agent_id].policy.act(obs[:, agent_id],
                                                                            render_rnn_states[:, agent_id],
                                                                            render_masks[:, agent_id],
                                                                            deterministic=True)

                        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                        render_actions[i, agent_id] = np.array(np.split(_t2n(render_action), self.n_render_rollout_threads))
                        render_rnn_states[i, agent_id] = np.array(np.split(_t2n(render_rnn_state), self.n_render_rollout_threads))

                render_actions_env = [render_actions[idx, :] for idx in range(self.n_render_rollout_threads)]

                print(render_actions_env[0], ready[0])

                # step
                combined_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)

                # Split the combined observations into obs and share_obs, then combine across environments.
                obs = []
                share_obs = []
                for o in combined_obs:
                    obs.append(o["obs"])
                    share_obs.append(o["share_obs"][0])
                obs = np.array(obs)
                share_obs = np.array(share_obs)

                # Pull agent ready state from the info message.
                for i in range(self.n_render_rollout_threads):
                    for a in range(self.num_agents):
                        ready[i, a] = render_infos[i]["ready"][a]

                # Display with ipython
                if not np.any(render_dones):
                    if ipython_clear_output:
                        clear_output(wait = True)
                    render_env.envs[0].env.render()

                # append frame
                if self.all_args.save_gifs:        
                    image = render_infos[0]["frame"]
                    frames.append(image)

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )

    @torch.no_grad()
    def compute(self):
        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                next_value = self.trainer[agent_id].policy.get_values(self.buffer[i][agent_id].share_obs[-1], 
                                                                    self.buffer[i][agent_id].rnn_states_critic[-1],
                                                                    self.buffer[i][agent_id].masks[-1])
                next_value = _t2n(next_value)
                self.buffer[i][agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = [[] for i in range(self.n_rollout_threads)]
        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_training()
                train_info = self.trainer[agent_id].train(self.buffer[i][agent_id])
                train_infos[i].append(train_info)       
                self.buffer[i][agent_id].after_update()

        return train_infos