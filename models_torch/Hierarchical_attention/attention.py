import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, params):
        super(Attention, self).__init__()

        self.params = params
        self.channel = params['encoder']['out_channels']
        self.hidden = params['decoder']['hidden_size']
        self.attention_dim = params['attention']['attention_dim']

        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.encoder_feature_conv = nn.Conv2d(self.channel, self.attention_dim, kernel_size=1)

        self.attention_conv = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    def forward(self, cnn_features, hidden, alpha_sum, image_mask, reprod_logger, i, name):

        query = self.hidden_weight(hidden)
        reprod_logger.add(f"alpha_sum_{i}_{name}_1", alpha_sum.cpu().detach().numpy())
        # reprod_logger.add(f"query_{i}", query.cpu().detach().numpy())
        alpha_sum_trans = self.attention_conv(alpha_sum)
        # reprod_logger.add(f"alpha_sum_trans_{i}", alpha_sum_trans.cpu().detach().numpy())
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0,2,3,1))
        # reprod_logger.add(f"coverage_alpha_{i}", coverage_alpha.cpu().detach().numpy())

        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        # reprod_logger.add(f"cnn_features_trans_{i}", cnn_features_trans.cpu().detach().numpy())

        alpha_score = torch.tanh(query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0,2,3,1))
        # reprod_logger.add(f"alpha_score_{i}", alpha_score.cpu().detach().numpy())
        energy = self.alpha_convert(alpha_score)
        # reprod_logger.add(f"energy_{i}_1", energy.cpu().detach().numpy())
        energy = energy - energy.max()
        # reprod_logger.add(f"energy_{i}_2", energy.cpu().detach().numpy())
        energy_exp = torch.exp(energy.squeeze(-1))
        # reprod_logger.add(f"energy_exp_{i}_1", energy_exp.cpu().detach().numpy())
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
            # reprod_logger.add(f"energy_exp_{i}_2", energy_exp.cpu().detach().numpy())
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        reprod_logger.add(f"alpha_{i}_{name}", alpha.cpu().detach().numpy())

        alpha_sum = alpha[:,None,:,:] + alpha_sum
        reprod_logger.add(f"alpha_sum_{i}_{name}_2", alpha_sum.cpu().detach().numpy())

        context_vector = (alpha[:,None,:,:] * cnn_features).sum(-1).sum(-1)
        # reprod_logger.add(f"context_vector_{i}", context_vector.cpu().detach().numpy())

        return context_vector, alpha, alpha_sum
