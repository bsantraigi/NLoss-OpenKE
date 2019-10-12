import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model


class TransESoftLoss(Model):
    def __init__(self, config):
        super(TransESoftLoss, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        # self.criterion = nn.MarginRankingLoss(self.config.margin, False)
        self.criterion = nn.LogSigmoid()
        self.gamma = 5
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform(self.rel_embeddings.weight.data)

    def _calc(self, h, t, r):
        return torch.norm(h + r - t, self.config.p_norm, -1)

    # def loss(self, p_score, n_score):
    #     if self.config.use_gpu:
    #         y = Variable(torch.Tensor([-1]).cuda())
    #     else:
    #         y = Variable(torch.Tensor([-1]))
    #     return self.criterion(p_score, n_score, y)

    def loss(self, score):
        return -torch.mean(self.criterion(score * self.batch_y))

    def forward(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        score = self.gamma - self._calc(h, t, r)

        return self.loss(score)

    def predict(self):
        h = self.ent_embeddings(self.batch_h)
        t = self.ent_embeddings(self.batch_t)
        r = self.rel_embeddings(self.batch_r)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()
