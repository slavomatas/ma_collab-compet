# Implementing a MADDPG Algorithm

## Summary

In this project i have used Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to solve the Tennis environment.

## Multi-Agent Deep Deterministic Policy Gradient

Traditional reinforcement learning approaches such as Q-Learning or policy gradient are poorly suited to multi-agent environments.
One issue is that each agent’s policy is changing as training progresses, and the environment becomes non-stationary from the perspective
of any individual agent (in a way that is not explainable by changes in the agent’s own policy). This presents learning stability challenges
and prevents the straightforward use of past experience replay, which is crucial for stabilizing deep Q-learning. Policy gradient methods,
on the other hand, usually exhibit very high variance when coordination of multiple agents is required.

The goal of Multi-Agent DDPG algorithm is to operate under under the following constraints:
1.) the learned policies can only use local information (i.e. their own observations) at execution time
2.) it does not assume a differentiable model of the environment dynamics
3.) it does not assume any particular structure on the communication method between agents

The Multi-Agent DDPG algorithm accomplishes the above mentioned constraints by adopting the framework of centralized training with decentralized execution.
The Multi-Agent DDPG algorithm is am extension of actor-critic policy gradient methods where the critic is augmented with extra information
about the policies of other agents.

A primary motivation behind MADDPG is that, if we know the actions taken by all agents, the environment is stationary even as the policies change,
since P(s <sup>'<\sup> | s; a<sub>1<\sub>,...., a<sub>N<\sub>, π<sub>1<\sub>,...., π<sub>N<\sub>) = P(s <sup>'<\sup> |s, a<sub>1<\sub>,….,a<sub>N<\sub>) = P(s<sup>`<\sup>|s; a<sub>1<\sub>,….,a<sub>N<\sub>;π<sub>1<\sub>,…..,π<sub>N<\sub>)
for any π<sub>i<\sub>= π<sup>`<\sup><sub>i<\sub>. This is not the case if we do not explicitly condition on the actions of other agents, as done for most traditional RL methods.

MADDPG is a multi-agent version of DDPG. DDPG is well suited to continuous control tasks and this just extends it to a multi-agent scenario.
More details can be found in the [MADDPG paper](https://arxiv.org/abs/1706.02275).

## Implementation

In MADDPG, each agent has it's own actor and it's own critic. Agents share a common experience replay buffer which contains tuples with states and actions from all agents.
Each agent does it's own sampling from shared replay buffer. This allows agents to learn their own reward function and incorporate the actions of other agents in their learning.
Therefore, we can train them in collaborative, competitive, or mixed environments.

The model consists of two separate networks for the actor and critic.

The actor is fully connected feed-forward network with two hidden layers, RELU activation and Batch Normalization.
The input is an state vector(dim=24), while the output is a vector with 2 values, one for each action.
The output actions are transformed with hyperbolic tangent non-linearity to squeeze the values to the -1..1 range.

```
    BatchNorm1d
      |
Fully Connected Layer (in=24 -> state size, out=1024)
      |
    RELU
    BatchNorm1d
      |
Fully Connected Layer (in=1024, out=300)
      |
    RELU
    BatchNorm1d
      |
Fully Connected Layer (in=300, out=2 -> action size)
      |
    tanh
```

The critic is fully connected feed-forward network with two hidden layers, RELU activation and Batch Normalization in the input layer.
The critic network estimates centralized action-value function that takes as input the observations of all agents x = (o<sub>1<\sub>,.....,o<sub>N<\sub>)
and the actions of all agents, a<sub>1<\sub>,.....,a<sub>N<\sub>, and outputs the Q-value for agent i.
Since each Q<sup>π<\sup><sub>i<\sub> is learned separately, agents can have arbitrary reward structures, including conflicting rewards in a competitive setting.

```
Fully Connected Layer (in=(24+2)*2 -> state size, out=512)
		  |
		RELU
		BatchNorm1d
		  |
Fully Connected Layer (in=(24+2)*2+512, out=300)
		  |
		RELU
		  |
Fully Connected Layer (in=300, out=1)
```


## Multi-Agent Deep Deterministic Policy Gradient Algorithm

```
for episode = 1 to M do
    Initialize a random OU process N for action exploration
    Receive initial state x
    for t = 1 to max-episode-length do
        for each agent i, select action a<sub>i<\sub> = µ<sub>θi<\sub>(o<sub>i<\sub>) + Nt w.r.t. the current policy and exploration
        Execute actions a = (a<sub>1<\sub>,.....,a<sub>N<\sub>) and observe reward r and new state x<sup>'<\sup>
        Store (x, a, r, x<sup>'<\sup>) in replay buffer D
        x<-x<sup>'<\sup>
        for agent i = 1 to N do
            Sample a random minibatch of S samples

            # ------------------------- Update critic by minimizing the loss ------------------------------- #
            next_states = torch.stack(next_states)
            next_actions_all = self.target_act(next_states)
            next_actions_all = torch.cat(next_actions_all, dim=1)

            next_states_all = torch.stack(next_states_all)
            with torch.no_grad():
                q_targets_next = agent.target_critic(next_states_all, next_actions_all)

            # Compute Q targets for current states (y_i)
            q_targets = torch.stack(rewards)[:, agent_number].view(-1, 1) + self.discount_factor * q_targets_next * (
                        1 - torch.stack(dones)[:, agent_number].view(-1, 1))

            # Compute Q expected
            states_all = torch.stack(states_all)
            actions_all = torch.stack(actions_all)
            # actions_all = torch.stack(actions_all).view(-1, self.action_size * self.agents)
            q_expected = agent.critic(states_all, actions_all)

            # Compute critic loss

            # huber_loss = torch.nn.SmoothL1Loss()
            # critic_loss = huber_loss(current_Q, target_Q.detach())
            critic_loss = F.mse_loss(q_expected, q_targets.detach())

            # Minimize the loss
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            # ---------------------------- Update actor using the sampled policy gradient---------------------------- #
            actions_pred = []
            for i in range(self.n_agents):
                if i == agent_number:
                    actions_pred.append(self.agents[i].actor(torch.stack(states)[:, i, :]))
                else:
                    actions_pred.append(self.agents[i].actor(torch.stack(states)[:, i, :]).detach())

            actions_pred = torch.cat(actions_pred, dim=1)

            # combine all the actions and observations for input to critic
            # many of the obs are redundant, and obs[1] contains all useful information already

            # get the policy gradient
            actor_loss = -agent.critic(states_all, actions_pred).mean()
            actor_loss.backward()

            # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
            agent.actor_optimizer.step()

        end for

        Update target network parameters for each agent i:
        θ<sup>'<\sup><sub>i<\sub> <- τθ<sub>i<\sub> + (1 − τ)θ<sup>'<\sup><sub>i<\sub>

    end for
end for
```

```

Agent hyperparameters:

```

BATCH_SIZE = 1024 # minibatch size

LR_ACTOR = 1e-2  # learning rate of the actor

LR_CRITIC = 1e-2 # learning rate of the critic

DISCOUNT = 0.99

TAU = 0.001 # for updating target networks

EPISODE_LENGTH = 500

STEPS_PER_UPDATE = 100 # update the network parameters after every 100 samples added to the replay buffer


### System Setup

 - Python:			3.6.6
 - CUDA:			  9.0
 - Pytorch: 		0.4.1
 - NVIDIA Driver:  	390.77
 - Conda:			4.5.10

 All training was performed on a single Ubuntu 18.04 desktop with an NVIDIA GTX 1080ti.

## Conclusion

The DDPG agent was able to train relative quickly compare to PPO agent. DDPG agent successfully solved environment in 252 episodes.
The PPO agent solved evironment in ~ 5000 episodes and as seen on the scores chart it was not as stable as expected.
I am trying to run run PPO again with different parameters, in particular LR_ACTOR is set now to 1e-5.

#### Key improvements
A couple critical changes were made, without which training was very poor:
  - L2 weight_decay of the critic set to 0.0 (vs 0.01 in the paper).  I think this is because the positive reward signal is pretty sparse in the beginning and this can easily get drowned out by the weight decay.
  - Actions concatenated into the input layer (vs the 1st hidden layer in the paper).  Not sure why this made a difference, I've seen others say it didn't change things much for them.

## Plot of Rewards
Using the supplied hyperparameters the agent is able to solve the environments in 2531 episodes.  See [here](assets/training_output.txt) for detailed training results.

This graph shows the average reward over the last 100 consecutive episodes (after taking the maximum over both agents).
![results](assets/results.png)


## Ideas for Future Work
  - It would be interesting to apply this algorithm to a more complex environment, like the 2 on 2 [SoccerTwos](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos) as this would involve agents working in both a collaborative and competitive manner simultaneously.  Given that each agent is able to learn it's own reward function, this should be feasible.


