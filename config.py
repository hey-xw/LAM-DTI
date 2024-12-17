# -*- coding:utf-8 -*-


class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 70
        self.Batch_size = 32
        self.Patience = 20
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.char_dim = 64

        self.loss_epsilon = 1
        self.prompt_len = 1100
        self.shared_dim = 256
        self.eps =  1e-9

        self.nheads =4
        self.n_levels =  3
        self.attn_dropout =  0.1
        self.relu_dropout =  0.0
        self.embed_dropout =  0.2
        self.res_dropout =  0.1
        self.attn_mask =  True


