from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from models.HRL import CandidatePolicyNetwork, WeightPolicyNetwork
import torch
import torch.nn.functional as F
import config.config as config


class EnsembleModel:
    def __init__(self, model_zoo, device, embed_dim=config.embedding_size):
        self.model_zoo = model_zoo
        self.num_models = len(model_zoo)

        self.device = device
        self.embed_dim = embed_dim

        self.performance_embedder = nn.Sequential(
            nn.Linear(self.num_models, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        ).to(self.device)

        # 初始化策略网络
        self.select_policy = CandidatePolicyNetwork(num_models=self.num_models).to(self.device)
        self.weight_policy = WeightPolicyNetwork(num_models=self.num_models).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.select_policy.parameters()) +
            list(self.weight_policy.parameters()) +
            list(self.performance_embedder.parameters()),
            lr=config.learning_rate,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def predict(self, x, prev_x, prev_y):
        x, prev_x, prev_y = x.to(self.device), prev_x.to(self.device), prev_y.to(self.device)

        prev_losses = []
        for model in self.model_zoo:
            with torch.no_grad():
                prev_pred = model.predict(prev_x)
                loss = F.mse_loss(prev_pred, prev_y, reduction='none')
                prev_losses.append(loss)
        prev_losses = torch.stack(prev_losses, dim=1)
        perf_embed = self.performance_embedder(prev_losses.squeeze(-1))

        encoded = self.select_policy.encoder(x)

        perf_embed_expand = perf_embed.unsqueeze(1).expand(-1, encoded.size(1), -1)
        concat_feat = torch.cat([encoded, perf_embed_expand], dim=-1)

        selected_mask = self.select_policy.select_models(concat_feat)
        weights, dist = self.weight_policy.assign_weights(concat_feat, selected_mask)

        all_preds = []
        for model in self.model_zoo:
            with torch.no_grad():
                pred = model.predict(x)
                all_preds.append(pred)

        all_preds = torch.stack(all_preds)

        final_pred = torch.sum(all_preds.permute(1, 0, 2) * weights.unsqueeze(2), dim=1)

        return final_pred, selected_mask, weights, dist

    def update_policies(self, x, y_true, selected_mask):
        device = self.device

        select_log_probs = self.select_policy.saved_log_probs[-1]
        weight_log_probs = self.weight_policy.saved_log_probs[-1]
        rewards = self.weight_policy.rewards[-select_log_probs.shape[0]:]

        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        current_loss = -rewards
        lambda_sparsity = config.lambda_sparsity
        selected_counts = selected_mask.sum(dim=1)
        sparsity_penalty = lambda_sparsity * selected_counts.float()

        with torch.no_grad():
            best_model_loss = torch.full((x.size(0),), float('inf'), device=device)
            masked_preds  = []

            mask_model_counts = torch.zeros(x.size(0), device=device)

            for i, model in enumerate(self.model_zoo):

                pred = model.predict(x.to(self.device))
                loss = F.mse_loss(pred, y_true, reduction='none').squeeze(1)
                best_model_loss = torch.minimum(best_model_loss, loss)

                mask = selected_mask[:, i]
                if mask.any():
                    masked_pred = pred * mask.float().view(-1, 1)
                    masked_preds.append(masked_pred)
                    mask_model_counts += mask.float()

            sum_pred = torch.stack(masked_preds).sum(dim=0)
            avg_pred = sum_pred / mask_model_counts.view(-1, 1)
            mask_avg_loss = F.mse_loss(avg_pred, y_true, reduction='none').squeeze(1)

        select_rewards = (best_model_loss - current_loss - sparsity_penalty).detach()

        weight_rewards = (mask_avg_loss - current_loss).detach().unsqueeze(1).expand_as(weight_log_probs)

        select_rewards = (select_rewards - select_rewards.mean()) / (select_rewards.std() + 1e-8)
        weight_rewards = (weight_rewards - weight_rewards.mean()) / (weight_rewards.std() + 1e-8)

        select_loss = - (select_log_probs * select_rewards.unsqueeze(1)).mean()
        weight_loss = - (weight_log_probs * weight_rewards).mean()

        loss = select_loss + weight_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.select_policy.saved_log_probs.clear()
        self.weight_policy.saved_log_probs.clear()
        self.weight_policy.rewards.clear()

    def train(self):
        self.select_policy.train()
        self.weight_policy.train()
        self.performance_embedder.train()

    def eval(self):
        self.select_policy.eval()
        self.weight_policy.eval()
        self.performance_embedder.eval()
