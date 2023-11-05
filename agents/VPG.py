
#
# Vanilla Policy Gradient (with GAE Lambda advantage estimation) agent
# modified for the AIQ test from the Spinning up pytorch implementation
#
# Copyright Joshua Achiam 2018 (OpenAI)
# Copyright Petr Zeman 2023
# Copyright Ondřej Vadinský 2023
# Copyright Jan Štipl 2023
# Released under GNU GPLv3
#

import sys
import time

import numpy as np
import torch
from torch.optim import Adam

from agents.utils.spinning_up_tools import mpi_pytorch, SpinCore as core, mpi_tools
from agents.utils.spinning_up_tools.PolicyEnvBuffer import PolicyEnvBuffer
from agents.utils.spinning_up_tools.logx import EpochLogger
from agents.utils.spinning_up_tools.run_utils import setup_logger_kwargs
from .Agent import Agent


class VPG(Agent):

    def __init__(self, refm, disc_rate, steps_per_epoch=40, train_v_iters=80, gamma=0.99, pi_lr=0.0003, vf_lr=0.001,
                 lam=0.97, hidden1=64, hidden2=64, hidden3=0, threads=1):
        Agent.__init__(self, refm, disc_rate)

        self.num_states = refm.getNumObs()  # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()
        self.obs_dim = refm.getNumObsSyms() * refm.getNumObsCells()
        self.act_dim = refm.getNumActions()
        self.steps_per_epoch = int(steps_per_epoch)
        self.train_v_iters = int(train_v_iters)
        self.epoch = 1
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.Lambda = lam
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.hidden3 = int(hidden3)
        self.threads = int(threads)
        self.logger_kwargs = dict()

        # if the internal discount rate isn't set, use the environment value
        if gamma == 0:
            self.gamma = disc_rate
        else:
            self.gamma = gamma

        if self.gamma >= 1.0:
            print("Error: VPG can only handle an internal discount rate ",
                  "that is below 1.0")
            sys.exit()

        self.error_file = str(refm) + "_" + str(disc_rate) + "_" + str(self.__str__()) + '_Exception_' + time.strftime(
            "_%Y_%m%d_%H_%M_%S", time.localtime()) + ".log"

        self.reset()

    # Reset of purely agent and Logging
    def setup_agent(self):

        # Reset state and action
        self.state = 0
        self.action = 0
        self.epoch_step = 0

        # Create actor-critic module
        actor_critic = core.MLPActorCritic
        if self.hidden3 == 0:
            self.ac = actor_critic(self.obs_dim, self.act_dim, hidden_sizes=(self.hidden1, self.hidden2))
        else:
            self.ac = actor_critic(self.obs_dim, self.act_dim, hidden_sizes=(self.hidden1, self.hidden2, self.hidden3))

        # Set up experience buffer
        self.local_steps_per_epoch = int(self.steps_per_epoch // mpi_tools.num_procs())
        self.buf = PolicyEnvBuffer(self.obs_dim, 1, self.local_steps_per_epoch, self.gamma, self.Lambda)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=self.vf_lr)

    def reset(self):

        # Prepare for interaction with environment
        self.start_time = time.time()
        self.logger_kwargs = setup_logger_kwargs(self.__str__(), data_dir='./policy-log/', datestamp=True)
        # Set up logger and save configuration
        self.logger = EpochLogger(output_fname=self.__str__() + '_' + '.csv', **self.logger_kwargs)

        # setup agent multithreading
        mpi_pytorch.setup_pytorch_for_mpi(self.threads)

        # self.logger.save_config(locals())

        self.setup_agent()
        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)

    def __str__(self):
        return "VPG(" + str(self.steps_per_epoch) + "," + str(self.train_v_iters) + "," + str(self.gamma) + "," + str(
            self.pi_lr) + "," + \
            str(self.vf_lr) + "," + str(self.Lambda) + "," + str(self.hidden1) + "," + str(self.hidden2) + "," + str(
                self.hidden3) + ")"

    # Set up function for computing VPG policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        act = act[:, -1]
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def log_update(self):
        # Log info about epoch
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('TotalEnvInteracts', (self.epoch + 1) * self.local_steps_per_epoch)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('Failed', self.failed)
        self.logger.log_tabular('Time', time.time() - self.start_time)
        self.logger.dump_tabular()

    def get_logs(self) -> dict:
        log_dict = dict()
        for i, log_data in enumerate(self.logger.get_full_log()):
            log_dict[i] = log_data
        return log_dict

    def update(self):
        data = self.buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        mpi_pytorch.mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_pytorch.mpi_avg_grads(self.ac.v)  # average grads across MPI processes
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                          KL=kl, Entropy=ent,
                          DeltaLossPi=(loss_pi.item() - pi_l_old),
                          DeltaLossV=(loss_v.item() - v_l_old))

        self.log_update()

    def perceive(self, observations, reward):

        if len(observations) != self.obs_cells:
            raise NameError("VPG received wrong number of observations!")

        # convert observations into a single number for the new state
        nstate = 0
        for i in range(self.obs_cells):
            nstate = observations[i] * self.obs_symbols ** i

        # convert new state into numpy array
        np_observation_list = np.array(nstate)

        # convert observation from numpy array through one hot encoding into tensor of dimension size
        np_observation_current: np.ndarray = np.eye(self.obs_dim)[np_observation_list.reshape(-1)]

        try:
            # Main loop: collect experience in env and update/log each epoch
            a, v, logp = self.ac.step(torch.as_tensor(np_observation_current, dtype=torch.float32))

            # save
            self.buf.store(np_observation_current, a.item(), reward, v, logp)
            self.logger.store(VVals=v)

            # Every epoch end process trajectory and update policy
            if self.epoch_step == self.local_steps_per_epoch - 1:
                self.logger.store(EpRet=reward, EpLen=self.local_steps_per_epoch)
                self.buf.finish_path(v)
                self.update()
                self.epoch_step = 0
                self.epoch += 1
            else:
                self.epoch_step += 1


        except Exception as _:
            self.failed = True
            self.setup_agent()
            return 0

        return a.item()

