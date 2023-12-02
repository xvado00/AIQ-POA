

POLICY OPTIMISATION AGENTS MODULE FOR THE AIQ TEST README
=========================================================


This is a module for the Python v3 AIQ Test (https://github.com/xvado00/AIQ)
that adds implementation of selected Policy Optimisation Agents:

- VPG.py  Vanilla Policy Gradient agent based on OpenAI SpinningUp:
https://spinningup.openai.com/en/latest/algorithms/vpg.html

- PPO.py  Proximal Policy Optimization agent based on OpenAI SpinningUp:
https://spinningup.openai.com/en/latest/algorithms/ppo.html

The module extends the original implementation that was part of a thesis:

Assessing Policy Optimization agents using Algorithmic IQ test
by Petr Zeman, Prague University of Economics and Business, 2023,

and was used for the experiments in paper:

Towards Evaluating Policy Optimisation Agents
using Algorithmic Intelligence Quotient Test
by Ondřej Vadinský and Petr Zeman, 2018.

The code is released under the GNU GPLv3.  See Licence.txt file.


Known Issues
------------

Both the agents experience NaN errors, most likely due to the advantages
between policy changes being almost 0 in some environments. The number
of errors increases with decreasing values of SPE hyperparameter. For
low values of SPE (10) this noticeably impacts agents performance.
The number of errors encountered by PPO is higher than that of VPG for
the same SPE setting.


Installation
------------

0) Install PyTorch and mpi4py.
1) Get the AIQ test from: https://github.com/xvado00/AIQ
2) Copy the contents of agents directory into the AIQ/agents.
3) Edit AIQ/agents/__init__.py to list the agents "VPG" and "PPO".


Usage
-----

Hyperparametrs for the VPG agent (and their defaults) are as follows:

- SPE: number of environment interaction steps per epoch,
- VFTI: number of gradient descent iterations
		for value function optimisation per epoch (80),
- gamma: discount factor (0.99),
- PLR: policy learning rate (0.0003),
- VFLR: value function learning rate (0.001),
- Lambda: balances variance (0) and bias (1) in
		generalised advantage estimation (0.97),
- hidden1: number of neurons in the 1st hidden layer of Actor and
		Critic networks (64),
- hidden2: number of neurons in the 2nd hidden layer of Actor and
		Critic networks (64),
- hidden3: number of neurons in the 3rd hidden layer of Actor and
		Critic networks, if set to 0, 3rd hidden layer is not used (0).

This resutls in the following agent string for the AIQ test:

VPG,<SPE>,80,0.99,0.0003,0.001,0.97,64,64,0

Suggested values for SPE: [30,150].

Hyperparametrs for the PPO agent (and their defaults) are as follows:

- SPE: number of environment interaction steps per epoch,
- PTI: maximum number of gradient ascent iterations
		for policy optimisation per epoch (80),
- VFTI: number of gradient descent iterations
		for value function optimisation per epoch (80),
- gamma: discount factor (0.99),
- PLR: policy learning rate (0.0003),
- VFLR: value function learning rate (0.001),
- Lambda: balances variance (0) and bias (1) in
		generalised advantage estimation (0.97),
- epsilon: clip ratio for new policy clipping (0.2),
- TKL: (approximate) KL-Divergence between new and old policy that
		triggers early-stopping of policy gradient update (0.01),
- hidden1: number of neurons in the 1st hidden layer of Actor and
		Critic networks (64),
- hidden2: number of neurons in the 2nd hidden layer of Actor and
		Critic networks (64),
- hidden3: number of neurons in the 3rd hidden layer of Actor and
		Critic networks, if set to 0, 3rd hidden layer is not used (0),

This resutls in the following agent string for the AIQ test:

PPO,<SPE>,80,80,0.99,0.0003,0.001,0.97,0.2,0.01,64,64,0

Suggested values for SPE: [30,250].


Refer to the AIQ Test Readme.txt for the test parameters and how to run it.

