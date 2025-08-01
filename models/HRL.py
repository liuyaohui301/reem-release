import torch
import torch.nn as nn
from torch.distributions import Beta
import config.config as config
from layers.EnhancedEncoder import EnhancedEncoder


class CandidatePolicyNetwork(nn.Module):
    def __init__(self, num_models, hidden_size=config.hidden_size, embed_size=config.embedding_size):
        super().__init__()
        self.num_models = num_models
        self.hidden_size = hidden_size
        self.encoder = EnhancedEncoder(hidden_size)

        self.policy_net = nn.Sequential(
            nn.Linear(config.seq_len * (hidden_size + embed_size), hidden_size + embed_size),
            nn.ReLU(),
            nn.Linear(hidden_size + embed_size, hidden_size + embed_size),
            nn.ReLU(),
            nn.Linear(hidden_size + embed_size, num_models),
            nn.Sigmoid()
        )

        self.saved_log_probs = []

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, encoded):
        batch_size = encoded.shape[0]
        flattened = encoded.view(batch_size, -1)
        probs = self.policy_net(flattened)

        return probs

    def select_models(self, encoded):
        probs = self.forward(encoded)

        dist = torch.distributions.Bernoulli(probs=probs)

        selected_mask = dist.sample()

        for i in range(selected_mask.size(0)):
            if selected_mask[i].sum() == 0:
                rand_idx = torch.randint(0, self.num_models, (1,)).to(selected_mask.device)
                selected_mask[i, rand_idx] = 1

        log_probs = dist.log_prob(selected_mask)

        self.saved_log_probs.append(log_probs)

        return selected_mask

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)


class WeightPolicyNetwork(nn.Module):
    def __init__(self, num_models, hidden_size=config.hidden_size, init_alpha=1.0, init_beta=1.0, embed_size=config.embedding_size):
        super().__init__()
        self.num_models = num_models
        self.hidden_size = hidden_size
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.encoder = EnhancedEncoder(hidden_size)

        self.mask_embedding = nn.Sequential(
            nn.Embedding(2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4)
        )

        self.joint_processing = nn.Sequential(
            nn.Linear(config.seq_len * (hidden_size + embed_size) + num_models * (hidden_size // 4), hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        self.policy_net_alpha = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_models),
            nn.Softplus()
        )

        self.policy_net_beta = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_models),
            nn.Softplus()
        )

        self._initialize_weights()

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, encoded, selected_mask):
        batch_size = encoded.shape[0]
        mask_embedded = self.mask_embedding(selected_mask.long())
        mask_flat = mask_embedded.view(batch_size, -1)
        x_flat = encoded.view(batch_size, -1)

        joint_input = torch.cat([x_flat, mask_flat], dim=-1)
        joint_features = self.joint_processing(joint_input)

        alpha = self.policy_net_alpha(joint_features).clamp(min=1e-3, max=1e3)
        beta = self.policy_net_beta(joint_features).clamp(min=1e-3, max=1e3)

        alpha = alpha * selected_mask.float() + 1e-3 * (~selected_mask.bool()).float()
        beta = beta * selected_mask.float() + 1e-3 * (~selected_mask.bool()).float()

        return alpha, beta

    def assign_weights(self, encoded, selected_mask):
        alpha, beta = self.forward(encoded, selected_mask)

        dist = Beta(alpha, beta)

        base_weights = dist.sample()

        masked_weights = base_weights * selected_mask.float()

        with torch.no_grad():
            weight_sums = masked_weights.sum(dim=1, keepdim=True)
            norm_weights = masked_weights / weight_sums.clamp(min=1e-6)

        log_probs = dist.log_prob(base_weights)

        log_probs = log_probs * selected_mask.float()

        self.saved_log_probs.append(log_probs)

        return norm_weights, dist

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        with torch.no_grad():
            alpha_final_layer = self.policy_net_alpha[-2]
            nn.init.constant_(alpha_final_layer.weight, 0.0)
            nn.init.constant_(alpha_final_layer.bias,
                              torch.as_tensor(self.init_alpha).log().item())

            beta_final_layer = self.policy_net_beta[-2]
            nn.init.constant_(beta_final_layer.weight, 0.0)
            nn.init.constant_(beta_final_layer.bias,
                              torch.as_tensor(self.init_beta).log().item())

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)
