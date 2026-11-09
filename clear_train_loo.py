#!/usr/bin/env python3
"""
clear_train_loo.py - CLEAR 移植第三件: PHEME18 9 折 event-LOO 训练.

官方 train_pheme.py 的模型与训练环逐字移植 (NewsFeaturePurification /
DualStreamFusionModel / ClassifyModel / MultiViewLoss), 两处等价改造:
  (1) 冻结 BERT (is_lock=True) 换成预计算 CLS 隐层 + 可训 pooler (clear_precompute),
      数学等价 (bert_pool 无 dropout, 确定性前向);
  (2) 官方 6:2:2 随机切分换成我们的 9 折 event-LOO (val = 训练事件分层 10%,
      RandomState(1000+折) — 与 MDFEND/MetaAdapt 移植同规).
超参逐字: batch 256, Adam(5e-5 / proto 5e-3), 150 epochs, seed 3407,
  MultiViewLoss(0.6, 0.826..., 0.1, hard_ratio=0.6, 0.07, eta1=0.2, eta2=0.01),
  OT 自 epoch 14, mining 自 epoch 40, L_E 类1权重 2.425, FM 类1权重 ep>5: 1.3.
LLM 先验 = 本地 Qwen3.5-9B (clear_gen_prior; 官方 GPT-4o-mini 的慷慨替换).
评论标签列 = 全零 (官方 Pheme.xlsx 同值, 模型解析不参与损失).

输出: results/clear_loo.json  逐折 {auc, f1_05, f1_tt, acc, best_ep} + macro, 断点续跑.
GPU <2GB (BERT 已离线), 每折 ~10-20 min.
"""
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, roc_auc_score

from probe_f1_audit import _f1, best_thr

CODE = Path(__file__).resolve().parent
CACHE = CODE / "results" / "clear_cache_hcls.pt"
OUT = CODE / "results" / "clear_loo.json"
BERT = "/home/hydro/code/models/bert-large-uncased"

BATCH = 256
MAX_EPOCH = 150
SEED = 3407


# ---------- 逐字: train_pheme.py (仅去掉日志) ----------
@torch.no_grad()
def sliced_w2_squared(x, y, num_proj=32):
    assert x.dim() == 2 and y.dim() == 2 and x.shape == y.shape
    B, D = x.shape
    dirs = torch.randn(num_proj, D, device=x.device, dtype=x.dtype)
    dirs = F.normalize(dirs, dim=1)
    x_proj, y_proj = x @ dirs.t(), y @ dirs.t()
    x_sort, _ = torch.sort(x_proj, dim=0)
    y_sort, _ = torch.sort(y_proj, dim=0)
    return (x_sort - y_sort).pow(2).mean()


class NewsFeaturePurification(nn.Module):
    def __init__(self, news_dim, num_classes, num_prototypes_per_class=1, hidden_dim=128):
        super().__init__()
        self.news_dim = news_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.ot_start_epoch = 14
        self._cur_epoch = 0
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(num_prototypes_per_class, news_dim))
            for _ in range(num_classes)])
        self.news_velocity_net = nn.Sequential(
            nn.Linear(news_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, news_dim))
        self.lambda_params = nn.Parameter(torch.full((num_classes,), 0.5))
        self.beta_params = nn.Parameter(torch.full((num_classes,), 2.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.input_norm = nn.LayerNorm(news_dim)

    def get_target_distribution(self, labels):
        batch_size = labels.size(0)
        z_target = torch.zeros(batch_size, self.news_dim, device=labels.device)
        for k in range(self.num_classes):
            mask = (labels == k)
            if mask.sum() > 0:
                prototype_indices = torch.randint(0, self.num_prototypes_per_class,
                                                   (mask.sum(),), device=labels.device)
                selected_prototypes = self.prototypes[k][prototype_indices]
                noise = torch.randn_like(selected_prototypes) * 0.1
                z_target[mask] = selected_prototypes + noise
        return z_target

    def set_epoch(self, epoch):
        self._cur_epoch = int(epoch)

    def get_news_velocity(self, z, t):
        if isinstance(t, (float, int)):
            t = torch.full((z.shape[0], 1), float(t), device=z.device)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        inp = torch.cat([z, t], dim=-1)
        return self.news_velocity_net(inp)

    def _ot_match_targets(self, z_raw, z_target, labels):
        z_raw_det, z_tar_det = z_raw.detach(), z_target.detach()
        labels_1d = labels.view(-1)
        z_target_perm = z_target.clone()
        for k in range(self.num_classes):
            idx = (labels_1d == k).nonzero(as_tuple=False).view(-1)
            n = int(idx.numel())
            if n <= 1:
                continue
            z0, z1 = z_raw_det[idx], z_tar_det[idx]
            with torch.no_grad():
                cost = torch.cdist(z0.float(), z1.float(), p=2).pow(2)
                cost_np = cost.cpu().numpy()
            if linear_sum_assignment is None:
                remaining = set(range(n))
                col_ind_list = [-1] * n
                for r in range(n):
                    best_c = min(remaining, key=lambda c: cost_np[r][c])
                    col_ind_list[r] = best_c
                    remaining.remove(best_c)
                col_ind = torch.tensor(col_ind_list, device=z_target.device, dtype=torch.long)
            else:
                _, col_ind_np = linear_sum_assignment(cost_np)
                col_ind = torch.tensor(col_ind_np, device=z_target.device, dtype=torch.long)
            z_target_perm[idx] = z_target[idx][col_ind]
        return z_target_perm

    def forward_train(self, z_raw, labels, return_stats=False, return_per_sample=False):
        z_raw = self.input_norm(z_raw)
        z_target = self.get_target_distribution(labels)
        if getattr(self, '_cur_epoch', 0) >= getattr(self, 'ot_start_epoch', 0):
            z_target = self._ot_match_targets(z_raw, z_target, labels)
        batch_size = z_raw.size(0)
        t = torch.rand(batch_size, 1, device=z_raw.device)
        z_t = (1 - t) * z_raw + t * z_target
        u_true = z_target - z_raw
        v_pred = self.get_news_velocity(z_t, t.squeeze(1))
        resid = v_pred - u_true
        loss_per_sample = (resid ** 2).mean(dim=1)
        labels_1d = labels.view(-1)
        label_w = torch.ones_like(loss_per_sample, dtype=loss_per_sample.dtype)
        cur_epoch = int(getattr(self, '_cur_epoch', 0))
        w1 = 1.0 if cur_epoch <= 5 else 1.3
        label_w[labels_1d == 1] = w1
        loss = torch.sum(loss_per_sample * label_w) / (torch.sum(label_w) + 1e-8)
        if (not return_stats) and (not return_per_sample):
            return loss
        if return_stats:
            with torch.no_grad():
                stats = {
                    "fm_mse": float(loss.detach().cpu()),
                    "u_norm": float(u_true.norm(dim=1).mean().detach().cpu()),
                    "v_norm": float(v_pred.norm(dim=1).mean().detach().cpu()),
                    "resid_norm": float(resid.norm(dim=1).mean().detach().cpu()),
                    "t_mean": float(t.mean().detach().cpu()),
                    "swd2_raw_target": float(sliced_w2_squared(
                        z_raw.detach(), z_target.detach(), num_proj=32).detach().cpu()),
                }
        else:
            stats = None
        if return_per_sample and return_stats:
            return loss, loss_per_sample, stats
        if return_per_sample:
            return loss, loss_per_sample
        return loss, stats

    def compute_heun_solver(self, z_raw):
        dt = 1.0
        v0 = self.get_news_velocity(z_raw, 0.0)
        z_guess = z_raw + v0 * dt
        v1 = self.get_news_velocity(z_guess, 1.0)
        z_pure = z_raw + (dt / 2) * (v0 + v1)
        kinetic_energy = (torch.mean(v0 ** 2, dim=-1) + torch.mean(v1 ** 2, dim=-1)) / 2
        return z_pure, kinetic_energy

    def compute_geometric_potential(self, z_pure):
        term_geo_list = []
        for k in range(self.num_classes):
            dists = torch.cdist(z_pure, self.prototypes[k])
            min_dist, _ = torch.min(dists, dim=1)
            mse_dist = (min_dist ** 2) / self.news_dim
            geo_potential = -F.softplus(self.lambda_params[k]) * mse_dist
            term_geo_list.append(geo_potential)
        return torch.stack(term_geo_list, dim=1)

    def prototype_class_mindist2(self, z_raw_norm):
        protos = []
        for k in range(self.num_classes):
            pk = self.prototypes[k]
            if pk.dim() == 1:
                pk = pk.unsqueeze(0)
            protos.append(pk)
        protos = torch.stack(protos, dim=0)
        z = z_raw_norm.unsqueeze(1).unsqueeze(2)
        dist2 = ((z - protos.unsqueeze(0)) ** 2).sum(dim=-1)
        return dist2.min(dim=-1).values

    def predict(self, z_news_input, return_stats=False):
        z_raw = self.input_norm(z_news_input)
        batch_size = z_raw.size(0)
        if not self.training:
            z_pure, kinetic_energy = self.compute_heun_solver(z_raw)
            term_geo = self.compute_geometric_potential(z_pure)
            term_kin = -F.softplus(self.gamma) * kinetic_energy.unsqueeze(1)
            term_kin = term_kin.expand(-1, self.num_classes)
        else:
            term_geo = self.compute_geometric_potential(z_raw)
            v0 = self.get_news_velocity(z_raw, 0.0)
            instant_energy = torch.mean(v0 ** 2, dim=-1)
            term_kin = -F.softplus(self.gamma) * instant_energy.unsqueeze(1).expand(-1, self.num_classes)
        term_beta = self.beta_params.unsqueeze(0).expand(batch_size, -1)
        final_logits = term_geo + term_kin + term_beta
        final_logits = torch.clamp(final_logits, min=-10.0, max=10.0)
        alpha = F.softplus(final_logits) + 1
        if return_stats:
            stats = {'_z_raw_norm': z_raw.detach()}
            if self.training:
                stats['_instant_energy'] = instant_energy.detach()
            else:
                stats['_instant_energy'] = kinetic_energy.detach()
            return alpha, stats
        return alpha


class DualStreamFusionModel(nn.Module):
    def __init__(self, news_dim, comment_dim, sentiment_counts_dim=8,
                 hidden_dim=128, num_heads=3, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.news_purifier = NewsFeaturePurification(
            news_dim=news_dim, num_classes=num_classes,
            num_prototypes_per_class=num_heads, hidden_dim=hidden_dim)

    def compute_aux_losses(self, z_news_input, z_comment_input, y_veracity, return_stats=False):
        if return_stats:
            loss_fm_news, fm_loss_per_sample, fm_stats = self.news_purifier.forward_train(
                z_news_input, y_veracity, return_stats=True, return_per_sample=True)
        else:
            fm_stats = None
            loss_fm_news, fm_loss_per_sample = self.news_purifier.forward_train(
                z_news_input, y_veracity, return_per_sample=True)
        prototypes = torch.cat([p.unsqueeze(0) for p in self.news_purifier.prototypes], dim=0)
        loss_proto = 0.0
        for k1 in range(self.num_classes):
            for k2 in range(k1 + 1, self.num_classes):
                p1 = F.normalize(prototypes[k1], dim=1)
                p2 = F.normalize(prototypes[k2], dim=1)
                similarity = torch.mm(p1, p2.t())
                loss_proto += torch.mean((similarity + 1) ** 2)
        loss_fm_senti = torch.tensor(0.0, device=z_news_input.device)
        if return_stats:
            return loss_fm_news, loss_proto, loss_fm_senti, fm_stats, fm_loss_per_sample
        return loss_fm_news, loss_proto, loss_fm_senti, fm_loss_per_sample

    def predict(self, z_news_input, z_comment_input=None, sentiment_target=None, return_stats=False):
        if return_stats:
            dirichlet_params, stats = self.news_purifier.predict(z_news_input, return_stats=True)
        else:
            dirichlet_params = self.news_purifier.predict(z_news_input)
            stats = None
        return (dirichlet_params, stats) if return_stats else dirichlet_params


class ClassifyModel(nn.Module):
    """官方版 + 唯一改造: bert_pool(词元) -> pooler(预计算 CLS). 其余逐字."""
    def __init__(self, hidden_size, pooler_state, news_labels=2, comments_labels=7):
        super().__init__()
        config_h = hidden_size
        self.pooler_dense = nn.Linear(config_h, config_h)      # HF BertPooler.dense
        self.pooler_dense.load_state_dict(pooler_state)
        self.dropout = nn.Dropout(0.1)
        self.comment_linear = nn.Linear(8 * config_h, 5 * config_h)
        self.dual_purifier = DualStreamFusionModel(
            news_dim=5 * config_h, comment_dim=5 * config_h,
            sentiment_counts_dim=comments_labels, hidden_dim=256,
            num_heads=3, num_classes=news_labels)
        self.contrastive_predictor = nn.Linear(5 * config_h, 4 * config_h)
        self.comments_labels = comments_labels

        def xav(r, c):
            return nn.Parameter(nn.init.xavier_uniform_(torch.ones(r, c)))
        self.w_gcb_p, self.w_gca_p = xav(config_h, config_h), xav(config_h, config_h)
        self.w_fcb_p, self.w_fca_p = xav(2 * config_h, config_h), xav(2 * config_h, config_h)
        self.w_gsb_p, self.w_gsa_p = xav(config_h, config_h), xav(config_h, config_h)
        self.w_fsb_p, self.w_fsa_p = xav(2 * config_h, config_h), xav(2 * config_h, config_h)
        self.w_tempp = xav(2 * config_h, config_h)
        self.batch_norm_temp0 = nn.BatchNorm1d(config_h)
        self.batch_norm_temp2 = nn.BatchNorm1d(config_h)
        self.batch_norm_tmp0 = nn.BatchNorm1d(config_h)
        self.batch_norm_tmp2 = nn.BatchNorm1d(config_h)
        self.w_gcb_c, self.w_gca_c = xav(config_h, config_h), xav(config_h, config_h)
        self.w_fcb_c, self.w_fca_c = xav(2 * config_h, config_h), xav(2 * config_h, config_h)
        self.w_gsb_c, self.w_gsa_c = xav(config_h, config_h), xav(config_h, config_h)
        self.w_fsb_c, self.w_fsa_c = xav(2 * config_h, config_h), xav(2 * config_h, config_h)
        self.w_tempc = xav(2 * config_h, config_h)
        self.batch_norm_temp0c = nn.BatchNorm1d(config_h)
        self.batch_norm_temp2c = nn.BatchNorm1d(config_h)
        self.batch_norm_tmp0c = nn.BatchNorm1d(config_h)
        self.batch_norm_tmp2c = nn.BatchNorm1d(config_h)
        self.w_r, self.w_z, self.w_h = (xav(4 * config_h, 2 * config_h) for _ in range(3))
        self.batch_norm_r = nn.BatchNorm1d(2 * config_h)
        self.batch_norm_z = nn.BatchNorm1d(2 * config_h)
        self.w_com1 = xav(4 * config_h, 2 * config_h)
        self.w_com2 = xav(4 * config_h, 2 * config_h)
        self.w_com3 = xav(4 * config_h, 2 * config_h)
        self.w_com4 = xav(4 * config_h, 2 * config_h)
        self.w_com5 = xav(4 * config_h, 2 * config_h)
        self.blbl_1 = xav(config_h, config_h // 2)
        self.blbl_2 = xav(config_h, config_h // 2)
        self.blbl_3 = xav(2 * config_h, config_h)
        self.blbl_4 = xav(2 * config_h, config_h)
        self.blbl_5 = xav(config_h, config_h // 2)
        self.blbl_6 = xav(config_h, config_h // 2)
        self.blbl_7 = xav(config_h, 2 * config_h)
        self.blbl_8 = xav(2 * config_h, config_h)
        self.blbl_7680 = xav(8 * config_h, config_h)
        self.batch_norm_blbl_1 = nn.BatchNorm1d(config_h)
        self.batch_norm_blbl_2 = nn.BatchNorm1d(2 * config_h)
        self.batch_norm_blbl_3 = nn.BatchNorm1d(config_h)
        self.batch_norm_blbl_4 = nn.BatchNorm1d(config_h)

    def bert_pool(self, h_cls):
        """官方 bert_pool = BertModel pooled = tanh(pooler.dense(CLS)); BERT 冻结已离线."""
        return torch.tanh(self.pooler_dense(h_cls))

    def HGD_FN_Post(self, message_1, message_2):
        phi_ci = F.leaky_relu(torch.cat(
            (torch.matmul(message_1, self.w_gcb_p), torch.matmul(message_2, self.w_gca_p)), dim=1),
            negative_slope=0.01)
        temp0 = self.batch_norm_temp0(torch.matmul(phi_ci, self.w_fcb_p))
        temp1 = torch.mul(message_1, temp0)
        temp2 = self.batch_norm_temp2(torch.matmul((1 - phi_ci), self.w_fca_p))
        temp3 = torch.mul(message_2, temp2)
        F_Ci = torch.tanh(temp1 + temp3)
        phi_si = F.leaky_relu(torch.cat(
            (torch.matmul(message_1, self.w_gsb_p), torch.matmul(message_2, self.w_gsa_p)), dim=1),
            negative_slope=0.01)
        tmp0 = self.batch_norm_tmp0(torch.matmul(phi_si, self.w_fsb_p))
        tmp1 = torch.mul(message_1, tmp0)
        tmp2 = self.batch_norm_tmp2(torch.matmul(phi_si, self.w_fsa_p))
        tmp3 = torch.mul(message_2, tmp2)
        F_Si = torch.tanh(tmp1 + tmp3)
        temppp = torch.matmul(phi_si, self.w_tempp)
        return torch.cat((F_Si, torch.mul(F_Ci, (1 - temppp))), dim=1)

    def HGD_FN_comment(self, com_n, com_n_n):
        phi_ci = F.leaky_relu(torch.cat(
            (torch.matmul(com_n, self.w_gcb_c), torch.matmul(com_n_n, self.w_gca_c)), dim=1),
            negative_slope=0.01)
        temp0 = self.batch_norm_temp0c(torch.matmul(phi_ci, self.w_fcb_c))
        temp1 = torch.mul(com_n, temp0)
        temp2 = self.batch_norm_temp2c(torch.matmul((1 - phi_ci), self.w_fca_c))
        temp3 = torch.mul(com_n_n, temp2)
        F_Ci = torch.tanh(temp1 + temp3)
        phi_si = F.leaky_relu(torch.cat(
            (torch.matmul(com_n, self.w_gsb_c), torch.matmul(com_n_n, self.w_gsa_c)), dim=1),
            negative_slope=0.01)
        tmp0 = self.batch_norm_tmp0c(torch.matmul(phi_si, self.w_fsb_c))
        tmp1 = torch.mul(com_n, tmp0)
        tmp2 = self.batch_norm_tmp2c(torch.matmul(phi_si, self.w_fsa_c))
        tmp3 = torch.mul(com_n_n, tmp2)
        F_Si = torch.tanh(tmp1 + tmp3)
        tempcp = torch.matmul(phi_si, self.w_tempc)
        return torch.cat((F_Si, torch.mul(F_Ci, (1 - tempcp))), dim=1)

    def SGRU(self, Xt, Ht_1):
        concatenated = torch.cat((Xt, Ht_1), dim=1)
        Rt = torch.sigmoid(self.batch_norm_r(torch.matmul(concatenated, self.w_r)))
        Zt = torch.sigmoid(self.batch_norm_z(torch.matmul(concatenated, self.w_z)))
        Ht_hidden = torch.tanh(torch.matmul(torch.cat((Xt, Rt * Ht_1), dim=1), self.w_h))
        return Zt * Ht_1 + (1 - Zt) * Ht_hidden

    def blbl(self, post, llm, HGD):
        p1 = torch.cat((torch.matmul(post, self.blbl_1), torch.matmul(llm, self.blbl_2)), dim=1)
        p2_1 = self.batch_norm_blbl_1(F.leaky_relu(torch.mul(p1, llm), negative_slope=0.01))
        p2_2 = torch.matmul(torch.cat((p1, llm), dim=1), self.blbl_3)
        p3 = self.batch_norm_blbl_2(torch.cat((torch.mul(p2_1, p2_2), llm), dim=1))
        H1 = torch.cat((HGD, llm), dim=1)
        H2 = torch.matmul(H1, self.blbl_4)
        H3_1 = torch.matmul(F.leaky_relu(H2, negative_slope=0.01), self.blbl_5)
        H3_2 = torch.matmul(torch.softmax(H2, dim=1), self.blbl_6)
        H3 = torch.cat((H3_1, H3_2), dim=1)
        H4_1 = self.batch_norm_blbl_3(torch.mul(post, H3))
        H4_1_1 = torch.mul(H1, torch.matmul(H4_1, self.blbl_7))
        H4_2 = self.batch_norm_blbl_4(torch.matmul(torch.cat((post, H3), dim=1), self.blbl_8))
        return torch.cat((H4_1_1, H4_2, p3), dim=1)

    def forward(self, h_cls, labels=None):
        f = [self.bert_pool(h_cls[:, i]) for i in range(17)]
        (pooled_post, pooled_com1, pooled_com11, pooled_com12, pooled_com13,
         pooled_com2, pooled_com21, pooled_com22, pooled_com23,
         pooled_com3, pooled_com31, pooled_com32, pooled_com33,
         pooled_com4, pooled_com41, pooled_com42, pooled_llm) = f
        contrast_p_1 = self.HGD_FN_Post(pooled_post, pooled_com1)
        contrast_p_2 = self.HGD_FN_Post(pooled_post, pooled_com2)
        contrast_p_3 = self.HGD_FN_Post(pooled_post, pooled_com3)
        contrast_p_4 = self.HGD_FN_Post(pooled_post, pooled_com4)
        contrast_c1_1 = self.HGD_FN_comment(pooled_com1, pooled_com11)
        contrast_c1_2 = self.HGD_FN_comment(pooled_post, pooled_com12)
        contrast_c1_3 = self.HGD_FN_comment(pooled_post, pooled_com13)
        Ht = self.SGRU(contrast_c1_2, contrast_c1_1)
        HT_1 = self.SGRU(contrast_c1_3, Ht)
        contrast_c2_1 = self.HGD_FN_comment(pooled_com2, pooled_com21)
        contrast_c2_2 = self.HGD_FN_comment(pooled_com2, pooled_com22)
        contrast_c2_3 = self.HGD_FN_comment(pooled_com2, pooled_com23)
        Ht = self.SGRU(contrast_c2_2, contrast_c2_1)
        HT_2 = self.SGRU(contrast_c2_3, Ht)
        contrast_c3_1 = self.HGD_FN_comment(pooled_com3, pooled_com31)
        contrast_c3_2 = self.HGD_FN_comment(pooled_com3, pooled_com32)
        contrast_c3_3 = self.HGD_FN_comment(pooled_com3, pooled_com33)
        Ht = self.SGRU(contrast_c3_2, contrast_c3_1)
        HT_3 = self.SGRU(contrast_c3_3, Ht)
        contrast_c4_1 = self.HGD_FN_comment(pooled_com4, pooled_com41)
        contrast_c4_2 = self.HGD_FN_comment(pooled_com4, pooled_com42)
        combine_4 = self.SGRU(contrast_c4_2, contrast_c4_1)
        combine_1 = torch.matmul(torch.cat((contrast_p_1, HT_1), dim=1), self.w_com1)
        combine_2 = torch.matmul(torch.cat((contrast_p_2, HT_2), dim=1), self.w_com2)
        combine_3 = torch.matmul(torch.cat((contrast_p_3, HT_3), dim=1), self.w_com3)
        HGD = torch.matmul(torch.cat((combine_1, combine_2, combine_3, combine_4), dim=1),
                           self.blbl_7680)
        Ending = self.blbl(pooled_post, pooled_llm, HGD)
        conments_end = torch.cat((HT_1, HT_2, HT_3, combine_4), dim=1)
        Ending_conmments = self.comment_linear(conments_end)
        z_news_input = self.dropout(Ending)
        aux_losses = {}
        if self.training and labels is not None:
            out_aux = self.dual_purifier.compute_aux_losses(z_news_input, None, labels)
            l_fm, l_proto, l_senti, fm_loss_per_sample = out_aux
            aux_losses['loss_fm_news'] = l_fm
            aux_losses['loss_fm_news_per_sample'] = fm_loss_per_sample
            aux_losses['loss_proto'] = l_proto
            aux_losses['loss_fm_senti'] = l_senti
            dirichlet_params, pred_stats = self.dual_purifier.predict(
                z_news_input, None, sentiment_target=None, return_stats=True)
            if pred_stats is not None:
                aux_losses.update(pred_stats)
        else:
            dirichlet_params = self.dual_purifier.predict(z_news_input, None)
        return dirichlet_params, aux_losses


class MultiViewLoss(nn.Module):
    """逐字 (去掉注释块与逐 batch 打印)."""
    def __init__(self, lambda_fused=0.5, lambda_m=0.2, lambda_t=0.1, hard_ratio=1.0,
                 temperature=0.07, eta1=1.0, eta2=0.01):
        super().__init__()
        self.lambda_fused = lambda_fused
        self.lambda_m = lambda_m
        self.lambda_t = lambda_t
        self.temperature = temperature
        self.hard_ratio = hard_ratio
        self.eta1 = eta1
        self.eta2 = eta2

    def compute_L_E(self, y, alpha, sample_weights=None):
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        digamma_sum_alpha = torch.digamma(sum_alpha)
        digamma_alpha = torch.digamma(alpha)
        term = y * (digamma_sum_alpha - digamma_alpha)
        L_E_per_sample = torch.sum(term, dim=1)
        true_labels = torch.argmax(y, dim=1)
        label_w = torch.ones_like(L_E_per_sample, dtype=torch.float)
        label_w[true_labels == 1] = 2.425
        if sample_weights is None:
            denom = torch.sum(label_w) + 1e-8
            return torch.sum(L_E_per_sample * label_w) / denom
        w = sample_weights.detach().float().view(-1)
        w = torch.clamp(w, min=0.0)
        total_w = w * label_w
        return torch.sum(L_E_per_sample * total_w) / (torch.sum(total_w) + 1e-8)

    def forward(self, epoch, dirichlet_params_fused, news_labels,
                comment_counts=None, aux_losses=None, eta3=0, sample_weights=None):
        y_onehot = F.one_hot(news_labels, num_classes=dirichlet_params_fused.size(1)).float()
        L_E = self.compute_L_E(y_onehot, dirichlet_params_fused, sample_weights=sample_weights)
        overall_loss = L_E * self.lambda_fused
        if aux_losses:
            if 'loss_fm_news' in aux_losses:
                loss_fm_news = aux_losses['loss_fm_news']
            elif 'loss_fm_news_per_sample' in aux_losses:
                loss_fm_news = aux_losses['loss_fm_news_per_sample'].mean()
            else:
                loss_fm_news = None
            if loss_fm_news is not None:
                overall_loss = overall_loss + self.eta1 * loss_fm_news
            if 'loss_proto' in aux_losses:
                overall_loss = overall_loss + self.eta2 * aux_losses['loss_proto']
            if 'loss_fm_senti' in aux_losses:
                overall_loss = overall_loss + eta3 * aux_losses['loss_fm_senti']
        return overall_loss


# ---------- 折循环 ----------
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mining_weights(model, dirichlet_params_fused, aux_losses, epoch):
    """逐字: 官方 Hard/Shift mining (epoch>=40, 只加权 L_EDL)."""
    if aux_losses is None or ('_z_raw_norm' not in aux_losses) \
            or ('_instant_energy' not in aux_losses) or epoch < 40:
        return None
    try:
        with torch.no_grad():
            z_raw_norm = aux_losses['_z_raw_norm']
            inst_energy = aux_losses['_instant_energy'].view(-1).float()
            alpha_det = dirichlet_params_fused.detach()
            p = alpha_det / (alpha_det.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=1).view(-1).float()
            dist2 = model.dual_purifier.news_purifier.prototype_class_mindist2(z_raw_norm)
            d_sorted, _ = torch.sort(dist2, dim=1)
            d1 = d_sorted[:, 0].view(-1).float()
            d2 = d_sorted[:, 1].view(-1).float()
            margin = (d2 - d1).clamp(min=0.0).view(-1).float()

            def _zscore(x):
                x = x.float().view(-1)
                return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)

            kappa = 0.5
            gap = _zscore(d1) + kappa * _zscore(inst_energy)
            inv_margin = -torch.log(margin + 1e-6)
            hard = _zscore(entropy) + _zscore(inv_margin)
            alpha_shift, alpha_hard = 0.03, 0.03
            w = torch.ones_like(entropy)
            w = w * (1.0 - alpha_shift * torch.sigmoid(gap))
            w = w * (1.0 - alpha_hard * torch.sigmoid(hard))
            w = w.clamp(0.94, 1.00)
            return w / (w.mean().detach() + 1e-8)
    except Exception as e:
        if not hasattr(model, "_mining_err_printed"):
            print(f"[Mining] ERROR -> disable weighting: {repr(e)}", flush=True)
            model._mining_err_printed = True
        return None


@torch.no_grad()
def run_scores(model, h, idx, device):
    model.eval()
    probs = []
    for i in range(0, len(idx), 512):
        alphas, _ = model(h[idx[i:i + 512]].to(device).float())
        probs.append((alphas / (alphas.sum(dim=1, keepdim=True) + 1e-12))[:, 1].cpu())
    return torch.cat(probs).numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nocmt', action='store_true',
                    help='无评论变体: 15 评论槽全置空串编码 (通道退化为常数偏置, 架构不动)')
    args = ap.parse_args()

    cache = torch.load(CACHE, map_location="cpu")
    h = cache["h"]                       # [5748, 17, 1024] fp16
    y = np.array(cache["label"])
    ev = np.array(cache["event"])
    events = sorted(set(ev.tolist()))
    device = "cuda"
    out_path = OUT.with_name("clear_loo_nocmt.json") if args.nocmt else OUT

    if args.nocmt:
        h = h.clone()
        h[:, 1:16] = cache["h_empty"].to(h.dtype).unsqueeze(0)

    from transformers import BertModel
    pooler_state = {k.replace("dense.", ""): v
                    for k, v in BertModel.from_pretrained(BERT).pooler.state_dict().items()}

    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.setdefault("folds", {})
    for fi, te_ev in enumerate(events):
        if te_ev in out["folds"]:
            print(f"[{te_ev}] cached", flush=True)
            continue
        seed_everything()
        te = np.where(ev == te_ev)[0]
        tr_all = np.where(ev != te_ev)[0]
        # val = 训练事件分层 10% (RandomState(1000+fi), 同 MDFEND/MetaAdapt 移植规)
        rng = np.random.RandomState(1000 + fi)
        val = []
        for c in (0, 1):
            idx = tr_all[y[tr_all] == c]
            val.extend(idx[rng.permutation(len(idx))[:max(1, int(len(idx) * 0.1))]])
        val = np.sort(np.array(val))
        tr = np.sort(np.setdiff1d(tr_all, val))
        if len(set(y[tr].tolist())) < 2:
            print(f"[{te_ev}] train 单类, 跳过", flush=True)
            continue
        # ebola-essien: te 全 label 1 — 主协议 (probe_fusesweep) 保留该折, auc 记 NaN, f1 照算
        te_single = len(set(y[te].tolist())) < 2

        model = ClassifyModel(h.shape[2], pooler_state).to(device)
        loss_fn = MultiViewLoss(lambda_fused=0.6, lambda_m=0.8260254001054855, lambda_t=0.1,
                                hard_ratio=0.6, temperature=0.07, eta1=0.2, eta2=0.01).to(device)
        proto_params = list(model.dual_purifier.news_purifier.prototypes.parameters())
        proto_ids = set(map(id, proto_params))
        base_params = [p for p in model.parameters() if id(p) not in proto_ids]
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 5e-5},
            {'params': proto_params, 'lr': 5e-3}])

        tr_h = h[tr].float()
        tr_y = torch.from_numpy(y[tr])
        best_val, best_state, best_ep = -1.0, None, 0
        t0 = time.time()
        for epoch in range(1, MAX_EPOCH + 1):
            model.train()
            model.dual_purifier.news_purifier.set_epoch(epoch)
            perm = torch.randperm(len(tr_h))
            for i in range(0, len(perm), BATCH):
                b = perm[i:i + BATCH]
                batch_h = tr_h[b].to(device)
                batch_y = tr_y[b].to(device)
                optimizer.zero_grad()
                dirichlet_params_fused, aux_losses = model(batch_h, batch_y)
                sample_weights = mining_weights(model, dirichlet_params_fused, aux_losses, epoch)
                loss = loss_fn(epoch, dirichlet_params_fused, batch_y,
                               comment_counts=None, aux_losses=aux_losses,
                               eta3=0, sample_weights=sample_weights)
                loss.backward()
                optimizer.step()
            val_p = run_scores(model, h, val, device)
            val_acc = accuracy_score(y[val], (val_p >= 0.5).astype(int))
            if val_acc > best_val:
                best_val, best_ep = val_acc, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == 1:
                print(f"  [{te_ev}] ep{epoch} val_acc={val_acc:.4f} "
                      f"(best {best_val:.4f}@{best_ep}, {time.time()-t0:.0f}s)", flush=True)

        model.load_state_dict(best_state)
        tr_s = run_scores(model, h, tr, device)
        te_s = run_scores(model, h, te, device)
        rec = {
            "auc": None if te_single else float(roc_auc_score(y[te], te_s)),
            "f1_05": _f1(te_s, y[te], 0.5),
            "f1_tt": _f1(te_s, y[te], best_thr(tr_s, y[tr])),
            "f1_or": _f1(te_s, y[te], best_thr(te_s, y[te])),
            "acc": float(accuracy_score(y[te], (te_s >= 0.5).astype(int))),
            "val_acc": float(best_val), "best_ep": best_ep, "n": int(len(te)),
            "secs": round(time.time() - t0)}
        out["folds"][te_ev] = rec
        out_path.write_text(json.dumps(out, indent=2))
        auc_s = f"{rec['auc']:.3f}" if rec["auc"] is not None else "NaN(单类te)"
        print(f"  [{te_ev}] auc={auc_s} f1_tt={rec['f1_tt']:.3f} "
              f"f1_05={rec['f1_05']:.3f} acc={rec['acc']:.3f} "
              f"(ep{best_ep}, {rec['secs']}s)", flush=True)

    done_folds = [e for e in events if e in out["folds"]]
    if len(done_folds) == len(events):
        TINY = ("ebola-essien", "gurlitt", "prince-toronto", "putinmissing")
        mac = {}
        for tag, sel in (("all", done_folds),
                         ("exclT", [e for e in done_folds if e not in TINY])):
            for k in ("auc", "f1_05", "f1_tt", "f1_or", "acc"):
                vals = [out["folds"][e][k] for e in sel
                        if out["folds"][e][k] is not None]
                if vals:
                    mac[f"{tag}_{k}"] = float(np.mean(vals))
        out["macro"] = mac
        OUT_note = f"(all: f1={mac.get('all_f1_tt', 0):.3f} auc={mac.get('all_auc', 0):.3f} | "
        OUT_note += f"exclT: f1={mac.get('exclT_f1_tt', 0):.3f} auc={mac.get('exclT_auc', 0):.3f})"
        print(f"== MACRO {OUT_note} ==", flush=True)
    out["variant"] = "nocmt" if args.nocmt else "full"
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
