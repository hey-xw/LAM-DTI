# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from AlignNets import AlignSubNet
from SubNets.transformers_encoder.transformer import TransformerEncoder
import torch.nn.functional as F


class LAMNet(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(LAMNet, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = 5

        # self.ctx_vectors = self._init_ctx(hp)
        # self.ctx = nn.Parameter(self.ctx_vectors)
        self.ctx = nn.Parameter(torch.zeros(hp.prompt_len, self.conv * 4))

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs+hp.prompt_len)

        self.ctx_max_pool = nn.MaxPool1d(hp.prompt_len)

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs+hp.prompt_len)

        # self.mix_attention_layer = nn.MultiheadAttention(
        #     self.attention_dim, self.mix_attention_head)

        #考虑清除到底要传哪些参数
        self.alignNet = AlignSubNet(hp, 'sim')

        self.embed_dim = self.conv * 4
        self.num_heads = hp.nheads
        self.layers = hp.n_levels
        self.attn_dropout = hp.attn_dropout
        self.relu_dropout = hp.relu_dropout
        self.res_dropout = hp.res_dropout
        self.embed_dropout = hp.embed_dropout
        self.attn_mask = hp.attn_mask

        self.trans_a_with_l = TransformerEncoder(embed_dim=self.embed_dim,
                                                 num_heads=self.num_heads,
                                                 layers=self.layers,
                                                 attn_dropout=self.attn_dropout,
                                                 relu_dropout=self.relu_dropout,
                                              res_dropout=self.res_dropout,
                                                 embed_dropout=self.embed_dropout,
                                                 attn_mask=self.attn_mask)
        self.gamma = nn.Parameter(torch.ones(self.embed_dim) * 1e-4)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        self.audio_proj = nn.Sequential(
            nn.LayerNorm(self.conv * 4),
            nn.Linear(self.conv * 4, self.conv * 4),
            nn.LayerNorm(self.conv * 4),
        )

        self.video_proj = nn.Sequential(
            nn.LayerNorm(self.conv * 4),
            nn.Linear(self.conv * 4, self.conv * 4),
            nn.LayerNorm(self.conv * 4),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.conv * 4),
            nn.Linear(self.conv * 4, self.conv * 4),
        )
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
    def _init_ctx(self,hp):
        ctx = torch.empty(hp.prompt_len, self.conv * 4, dtype=torch.float)
        # nn.init.trunc_normal_(ctx)
        self.custom_positive_trunc_normal_(ctx)
        return ctx

    def custom_positive_trunc_normal_(self,ctx,mean=0., std=1., min_val=0.):
        # 生成截断的正态分布样本
        samples = torch.normal(mean, std, ctx.size())
        # 只保留正数
        ctx.copy_(samples.clamp(min=min_val))


    def forward(self, drug, protein):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)


        # [B, D_C, F_0] -> [B, F_0, D_C]
        # [B, D_C, T_0] -> [B, D_0, D_C]
        drugConv = drugConv.permute(0,2,1)
        proteinConv = proteinConv.permute(0,2,1)
        # generate and employ modality-aware prompt
        batch_ctx = self.ctx.unsqueeze(0).repeat(drugConv.shape[0], 1, 1)
        _, aligned_drugConv, aligned_proteinConv  = self.alignNet(batch_ctx, drugConv, proteinConv)
        # _, aligned_drugConv, aligned_proteinConv  = self.alignNet(drugConv, drugConv, proteinConv)
        aligned_drug = self.audio_proj(aligned_drugConv)
        aligned_protein = self.video_proj(aligned_proteinConv)

        batch_ctx = self.text_proj(batch_ctx)
        # generated_ctx = self.trans_a_with_l(batch_ctx.permute(1, 0, 2), aligned_drugConv.permute(1, 0, 2), aligned_proteinConv.permute(1, 0, 2)).permute(1, 0, 2)
        # generated_ctx = batch_ctx + self.out_proj(generated_ctx) * self.gamma
        # for i in range(embedding_output.shape[0]):
        #     embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = generated_ctx[i]
        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        # drug_QKV = drugConv.permute(2, 0, 1)
        # protein_QKV = proteinConv.permute(2, 0, 1)
        #
        # # cross Attention
        # # [F_C, B, D_C] -> [F_C, B, D_C]
        # # [T_C, B, D_C] -> [T_C, B, D_C]
        # drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        # protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)
        #
        # # [F_C, B, D_C] -> [B, D_C, F_C]
        # # [T_C, B, D_C] -> [B, D_C, T_C]
        # drug_att = drug_att.permute(1, 2, 0)
        # protein_att = protein_att.permute(1, 2, 0)

        # drugConv = drugConv * 0.5 + drug_att * 0.5
        # proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugConv = torch.cat([drugConv,aligned_drug],dim=1)
        proteinConv = torch.cat([proteinConv,aligned_protein],dim=1)

        drugConv = self.Drug_max_pool(drugConv.permute(0,2,1)).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv.permute(0,2,1)).squeeze(2)

        #直接池化生成的prompt
        # ctx = generated_ctx.permute(0,2,1)
        # pair = self.ctx_max_pool(ctx).squeeze(2)
        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

