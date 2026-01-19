import torch
import torch.nn as nn
import torch.nn.functional as F

def create_simple_network(input_dim, output_dim, hidden_dim, num_hidden_layers=0, final_activation=None):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    if final_activation is not None:
        layers.append(final_activation())
    
    return nn.Sequential(*layers)

class CLUBSample_group(nn.Module):  # Sampled version of the CLUB estimator ICML2020 paper
    def __init__(self, x_dim, y_dim, hidden_size, num_hidden_layers=1):
        super(CLUBSample_group, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.p_mu = create_simple_network(
            input_dim=self.x_dim,
            output_dim=self.y_dim,
            hidden_dim=self.hidden_size//2,
            num_hidden_layers=self.num_hidden_layers,
            final_activation=None
        )

        self.p_logvar = create_simple_network(
            input_dim=self.x_dim,
            output_dim=self.y_dim,
            hidden_dim=self.hidden_size//2,
            num_hidden_layers=self.num_hidden_layers,
            final_activation=nn.Tanh
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)  # mu/logvar: (bs, y_dim)
        # mu = mu.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, mu.shape[
        #     -1])  # (bs, y_dim) -> (bs, 1, y_dim) -> (bs, T, y_dim) -> (bs*T, y_dim)
        # mu = mu.reshape(-1, mu.shape[-1])
        # logvar = logvar.unsqueeze(1).expand(-1, y_samples.shape[1], -1).reshape(-1, logvar.shape[-1])
        # logvar = logvar.reshape(-1, logvar.shape[-1])
        # y_samples = y_samples.reshape(-1, y_samples.shape[-1])  # (bs, T, y_dim) -> (bs*T, y_dim)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def mi_est(self, x_samples, y_samples):  # x_samples: (bs, x_dim); y_samples: (bs, y_dim)

        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()

        positive = - (mu - y_samples) ** 2 / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

        return upper_bound / 2
    


