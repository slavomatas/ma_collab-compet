import torch as th
import torch.nn as nn
import torch.nn.functional as F


'''
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = f.relu  # leaky_relu
        self.actor = actor
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # return a vector of the force
            h1 = self.nonlin(self.fc1(x))

            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            norm = torch.norm(h3)

            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            return 10.0 * (f.tanh(norm)) * h3 / norm if norm > 0 else 10 * h3

        else:
            # critic network simply outputs a number
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3
'''


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(state_size, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, action_size)

    # action output between -2 and 2
    def forward(self, state):
        result = F.relu(self.FC1(state))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result


class Critic(nn.Module):
    def __init__(self, state_size, action_size, agents, fc1_units=1024, fc2_units=512, fc3_units=300):
        super(Critic, self).__init__()
        self.agents = agents

        self.FC1 = nn.Linear(state_size * agents, fc1_units)
        self.FC2 = nn.Linear(1024+action_size * agents, fc2_units)
        self.FC3 = nn.Linear(fc2_units, fc3_units)
        self.FC4 = nn.Linear(fc3_units, 1)

    # obs: batch_size * state_size
    def forward(self, state, action):
        result = F.relu(self.FC1(state))
        combined = th.cat([result, action], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


