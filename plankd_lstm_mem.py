import math
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import numpy as np
import logging
from torchvision import transforms
from timm.models.resnet2 import resnet6e, resnet10_2, resnet6d
from torch.autograd import Variable
import torch.nn.init as init
import cv2

import matplotlib.cm as cm

import os

import matplotlib.pyplot as plt
from PIL import Image

torch.autograd.set_detect_anomaly(True)




class WP_Attention(nn.Module):
    def __init__(
        self,
        device,
        args
    ):
        super().__init__()
        self.device=device
        self.sigma = 3.0 
        self.bev_encoder = resnet6e(pretrained = False)
        self.args = args
        self.beta_r = self.args.beta_r
        
        self.wp_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            
        )
        
        self.query_encoder = nn.Sequential(
            nn.Linear(7 + 256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
        )
        
        self.W_q = nn.Linear(256, 128).to(self.device)
        self.W_k = nn.Linear(64, 128).to(self.device)
        
        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def forward(self, wp, bev, measurement, actor_pos, actor_cnt):
        
        x = torch.linspace(-18, 18, 180)
        y = torch.linspace(-18, 18, 180)
        #xx, yy = torch.meshgrid(x, y)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([xx, yy]).to(self.device)
        coords = coords.repeat((bev.shape[0], 1, 1, 1))
        bev =  torch.cat((bev, coords), dim=1)
        
        bev, _ = self.bev_encoder(bev) 
        
        query = torch.cat((bev, measurement), dim = 1)
        query = self.query_encoder(query).unsqueeze(1)
        query = self.W_q(query)
            
        key = self.wp_encoder(wp)
        key = self.W_k(key)
        
        score = torch.matmul(query, key.transpose(1, 2))  
        score = score.squeeze(1) / torch.sqrt(torch.tensor(128)) 
        
        # waypoints: (batch_size, num_waypoints, 2)
        # reference_points: (batch_size, max_num_reference_points, 2)
        
        pairwise_dists = torch.cdist(wp.float(), actor_pos.float(), p=2) # shape: (batch_size, num_waypoints, max_num_reference_points)
        kernel_vals = torch.exp(- pairwise_dists ** 2 / (2 * self.sigma ** 2)) # shape: (batch_size, num_waypoints, max_num_reference_points)
        mask = (actor_pos.sum(dim=-1) != 0).unsqueeze(1)  # shape: (batch_size, 1, max_num_reference_points)
        kernel_vals *= mask.float()
        norm_vals = torch.sum(kernel_vals, dim=-1)  # shape: (batch_size, num_waypoints)
        
        score = torch.softmax(score, dim = 1)
        
        diff = norm_vals.unsqueeze(dim=2) - norm_vals.unsqueeze(dim=1)
        labels = torch.sign(diff)
        labels_flat = labels.flatten()
        
        score_diff = score.unsqueeze(dim=2) - score.unsqueeze(dim=1)
        score_flat = score_diff.flatten()

        
        rank_loss = torch.max(torch.zeros(len(score_flat)).to(self.device), -labels_flat * score_flat + 0.1).mean()
        
        
        entropy = torch.mean(torch.sum(score * torch.log(score + 1e-8), dim=1))
        
        
        
        return score, entropy, rank_loss
    

    
    
    
class IB_Discriminator(nn.Module):
    def __init__(
        self,
        device,
    ):
        super().__init__()
        self.device=device
        
        self.main = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        )
        
        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        

    def forward(self, z):
        
        z = self.main(z)

        return z
    
    
class VIB(nn.Module):
    def __init__(
        self,
        device,
        base_model,
    ):
        super().__init__()
        self.device = device
        self.base_model = base_model 
        self.beta = 1e-3
        
        if self.base_model == 'interfuser':
            self.init_dim = 1296
        
        self.encoder = nn.Sequential(
            nn.Linear(self.init_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 3 + 2 * 7), 
        )
        
        
        self.pred_list = ['light_state', 'junction_state', 'stop_sign', 'is_vehicle_present', 
                          'is_pedestrian_present', 'brake', 'steer', 'throttle']
        
        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        
    def reparametrize_n(self, mu, std, n=1):

        eps = Variable(std.data.new(std.size()).normal_().to(self.device))
        return mu + eps * std

    def forward_encoder(self, feature_list):
        
        all_feature = None
        for i in range(len(feature_list)):
            feature = torch.mean(feature_list[i], dim = 1)
            feature = torch.flatten(feature, 1)
            if all_feature is None:
                all_feature = feature
            else:
                all_feature = torch.cat((all_feature, feature), dim=1)
        
        statistics  = self.encoder(all_feature)
        mu = statistics[:, :256]
        std = F.softplus(statistics[:, 256:] - 5, beta = 1)
        
        z = self.reparametrize_n(mu, std)
        
        return mu, std, z   
    
    def get_gussian_prob(self, mu, std, z):

        pi = torch.tensor([3.141592653589793]).to(self.device)
        coef = 1 / (torch.sqrt(2 * pi * torch.clamp(std,1e-15,100)**2) + 1e-15)
        exp = torch.exp(-((z - mu)**2) / ((2 * std**2) + 1e-15))
        pdf = coef * exp
        pdf = torch.clamp(pdf, 1e-15, 1)
        if (pdf <= 0).any():
            import pdb
            pdb.set_trace()
        log_pdf = torch.log(pdf)
        log_pdf = log_pdf.mean()
        return log_pdf
    
    def get_loss(self, mu, std, logit, x):
        
        class_loss = 0
        st = 0
        ed = 0
        for p in self.pred_list:
            if p == 'light_state':
                ed += 3
            else:
                ed += 2
            
            #class_loss += F.cross_entropy(logit[:, st:ed], torch.tensor(x[p])).div(math.log(2))    
            class_loss += F.cross_entropy(logit[:, st:ed], x[p].clone().detach()).div(math.log(2))
            if p == 'light_state':
                st += 3
            else:
                st += 2
        
        
        info_loss = -0.5*(1+2*(std + 1e-15).log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))              
        
        total_loss = class_loss + self.beta*info_loss
        
        return total_loss
    
    def forward(self, feature_list, x):
        
        mu, std, z = self.forward_encoder(feature_list)
        
        logit = self.decoder(z)
        
        total_loss = self.get_loss(mu, std, logit, x)

        return total_loss


class WpCombinedLoss(nn.Module):
    def __init__(self, device, alpha=0.5):

        super(WpCombinedLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        # LSTM 模块，用于提取全局特征
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=1, batch_first=True).to(self.device)
        # 全连接层，将 LSTM 的隐藏状态映射到固定维度
        self.fc = nn.Linear(128, 64).to(self.device)
        # 注意力层，用于计算每个时间步的重要性权重
        self.attention = nn.Linear(128, 1).to(self.device)
        self.mse_loss = nn.MSELoss(reduction='none').to(self.device)
    def forward(self, wp_s, wp_t):
        wp_t = wp_t.detach()

        wp_s = wp_s.to(self.device)
        wp_t = wp_t.to(self.device)
        h_s, _ = self.lstm(wp_s)  # [batch_size, seq_len=10, hidden_size=128]
        h_t, _ = self.lstm(wp_t)  # [batch_size, seq_len=10, hidden_size=128]
        attn_scores_s = self.attention(h_s)  # [batch_size, seq_len=10, 1]
        attn_weights_s = torch.softmax(attn_scores_s, dim=1)  # [batch_size, seq_len=10, 1]

        attn_scores_t = self.attention(h_t)  # [batch_size, seq_len=10, 1]
        attn_weights_t = torch.softmax(attn_scores_t, dim=1)  # [batch_size, seq_len=10, 1]

        context_s = torch.sum(h_s * attn_weights_s, dim=1)  # [batch_size, hidden_size=128]
        context_t = torch.sum(h_t * attn_weights_t, dim=1)  # [batch_size, hidden_size=128]

        feat_s = self.fc(context_s)  # [batch_size, 64]
        feat_t = self.fc(context_t)  # [batch_size, 64]

        global_loss = self.mse_loss(feat_s, feat_t)  # [batch_size, 64]
        global_loss = global_loss.mean(dim=1)  # [batch_size]

        local_loss = self.mse_loss(wp_s, wp_t)  # [batch_size, seq_len=10, 2]
        local_loss = local_loss.mean(dim=2).mean(dim=1)  # [batch_size]

        loss = self.alpha * global_loss + (1 - self.alpha) * local_loss  # [batch_size]
        loss = loss.mean()  # 标量
        return loss  # 返回每条轨迹的综合损失
    
    
class HierarchicalFeatureAlignment(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # 学生模型的投影层
        self.projectors_s = nn.ModuleDict({
            'early': nn.Sequential(
                nn.Linear(64, 256),  # 学生模型 early 层输出为64
                nn.LayerNorm(256)
            ).to(device),
            'mid': nn.Sequential(
                nn.Linear(128, 256),  # 学生模型 mid 层输出为512
                nn.LayerNorm(256)
            ).to(device),
            'late': nn.Sequential(
                nn.Linear(512, 256),  # 学生模型 late 层输出为2048
                nn.LayerNorm(256)
            ).to(device)
        })
        # 教师模型的投影层
        self.projectors_t = nn.ModuleDict({
            'early': nn.Sequential(
                nn.Linear(256, 256),  # 教师模型 early 层输出为256
                nn.LayerNorm(256)
            ).to(device),
            'mid': nn.Sequential(
                nn.Linear(512, 256),  # 教师模型 mid 层输出为512
                nn.LayerNorm(256)
            ).to(device),
            'late': nn.Sequential(
                nn.Linear(2048, 256),  # 教师模型 late 层输出为2048
                nn.LayerNorm(256)
            ).to(device)
        })

    def forward(self, s_feat, t_feat, level):
        # 全局平均池化
        s_feat = torch.mean(s_feat, dim=[2, 3])
        t_feat = torch.mean(t_feat, dim=[2, 3])
        
        # 打印特征形状用于调试
        # print(f"Level {level} - s_feat shape after pooling: {s_feat.shape}")
        # print(f"Level {level} - t_feat shape after pooling: {t_feat.shape}")
        
        # 断言特征维度
        expected_dims_s = {'early': 64, 'mid': 128, 'late': 512}
        expected_dims_t = {'early': 256, 'mid': 512, 'late': 2048}
        assert s_feat.size(1) == expected_dims_s[level], f"学生模型的{level}层期望特征维度为{expected_dims_s[level]}，但得到{s_feat.size(1)}"
        assert t_feat.size(1) == expected_dims_t[level], f"教师模型的{level}层期望特征维度为{expected_dims_t[level]}，但得到{t_feat.size(1)}"
        
        # 投影特征
        s_feat = self.projectors_s[level](s_feat)
        t_feat = self.projectors_t[level](t_feat)
        
        # 返回MSE损失
        return F.mse_loss(s_feat, t_feat)
class EncoderDecoderAlignment(nn.Module):
    def __init__(self, s_embed_dim, t_embed_dim):
        super().__init__()
        # 学生模型的投影层
        self.encoder_proj_s = nn.Sequential(
            nn.Linear(s_embed_dim, s_embed_dim),
            nn.LayerNorm(s_embed_dim)
        )
        self.decoder_proj_s = nn.Sequential(
            nn.Linear(s_embed_dim, s_embed_dim),
            nn.LayerNorm(s_embed_dim)
        )
        # 教师模型的投影层
        self.encoder_proj_t = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.LayerNorm(t_embed_dim)
        )
        self.decoder_proj_t = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.LayerNorm(t_embed_dim)
        )
        # 为匹配维度，添加一个线性层将教师特征映射到学生特征维度
        self.match_proj = nn.Linear(t_embed_dim, s_embed_dim)

    def forward(self, s_memory, t_memory, s_hs, t_hs):
        # s_memory 和 t_memory 形状：[seq_len, batch_size, embed_dim]
        # s_hs 和 t_hs 形状：[batch_size, num_queries, embed_dim]
        
        # 对 encoder features 进行平均
        s_memory_avg = s_memory.mean(dim=0)  # [batch_size, s_embed_dim]
        t_memory_avg = t_memory.mean(dim=0)  # [batch_size, t_embed_dim]
        
        # 对 decoder features 进行平均
        s_hs_avg = s_hs.mean(dim=1)  # [batch_size, s_embed_dim]
        t_hs_avg = t_hs.mean(dim=1)  # [batch_size, t_embed_dim]
        
        # 投影
        s_memory_proj = self.encoder_proj_s(s_memory_avg)
        t_memory_proj = self.encoder_proj_t(t_memory_avg)
        t_memory_proj = self.match_proj(t_memory_proj)  # 将教师特征映射到学生特征维度

        s_hs_proj = self.decoder_proj_s(s_hs_avg)
        t_hs_proj = self.decoder_proj_t(t_hs_avg)
        t_hs_proj = self.match_proj(t_hs_proj)  # 将教师特征映射到学生特征维度
        
        # 计算 MSE 损失
        enc_loss = F.mse_loss(s_memory_proj, t_memory_proj)
        dec_loss = F.mse_loss(s_hs_proj, t_hs_proj)
        
        return enc_loss, dec_loss
class PlanKD(nn.Module):
    def __init__(
        self,
        device,
        teacher_model, 
        student_model,
        base_model,
        args,

    ):
        super().__init__()
        self.device=device
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.args = args
        
        if base_model[:10] == 'interfuser':
            self.base_model = 'interfuser'
            
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad_(False)
        self.feature_alignment = HierarchicalFeatureAlignment(device)
        s_embed_dim = student_model.embed_dim  # 假设学生模型有 embed_dim 属性
        t_embed_dim = teacher_model.embed_dim  # 假设教师模型有 embed_dim 属性
        print(f"Student embed_dim: {s_embed_dim}")
        print(f"Teacher embed_dim: {t_embed_dim}")
        # 添加encoder-decoder alignment模块
        self.enc_dec_alignment = EncoderDecoderAlignment(s_embed_dim, t_embed_dim).to(device)
        self.IB_discriminator =  IB_Discriminator(device = self.device)
        self.IB = VIB(device = self.device, base_model = self.base_model)
        self.wp_attention = WP_Attention(device = self.device, args = self.args)
    
        self.IB_optimizer = torch.optim.Adam(self.IB.parameters(), lr = 1e-4)
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.bce_loss = nn.BCELoss(reduction='mean').to(self.device)

        self.wp_lstm_loss = WpCombinedLoss(device=self.device)  # 使用新的 LSTM Loss 类

        self.cam_exits = 0
        

    
    def train_IB(self, loss):
        
        self.IB_optimizer.zero_grad()
        loss.backward()
        self.IB_optimizer.step()
        
        return
    
    def get_cnn_features(self, backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        if 'act1' in dir(backbone):
            x = backbone.act1(x)
        else:
            x = backbone.relu(x)
        x = backbone.maxpool(x)

        x_layer1 = backbone.layer1(x)
        x_layer2 = backbone.layer2(x_layer1)
        # x_layer3 = backbone.layer3(x_layer2)
        # x_layer4 = backbone.layer4(x_layer3)
        return x_layer2
    
    def get_cnn_features_last(self, backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        if 'act1' in dir(backbone):
            x = backbone.act1(x)
        else:
            x = backbone.relu(x)
        x = backbone.maxpool(x)

        x_layer1 = backbone.layer1(x)
        x_layer2 = backbone.layer2(x_layer1)
        x_layer3 = backbone.layer3(x_layer2)
        x_layer4 = backbone.layer4(x_layer3)
        return x_layer3
    
    def get_hierarchical_features(self, backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.act1(x) if hasattr(backbone, 'act1') else backbone.relu(x)
        x = backbone.maxpool(x)

        early = backbone.layer1(x)     # early features
        mid = backbone.layer2(early)   # mid features  
        late = backbone.layer4(backbone.layer3(mid))  # late features

        return {
            'early': early,
            'mid': mid,
            'late': late
        }



    def forward(self, x):
        
        
        output_s = self.student_model(x)
        output_t =self.teacher_model(x)
        
        # 获取学生模型和教师模型的路点预测
        wp_s = output_s[1]  # 学生模型的路点预测，形状：[batch_size, seq_len, 2]
        wp_t = output_t[1].detach()  # 教师模型的路点预测，使用 detach() 防止梯度回传

        # 使用 WpLSTMLoss 计算路点损失
        #wp_g_loss = torch.tensor(0)
        wp_g_loss = self.wp_lstm_loss(wp_s, wp_t)
        if self.base_model == 'interfuser':
            feature_t_front = self.get_cnn_features(self.teacher_model.rgb_backbone, x['rgb'])
            feature_t_left = self.get_cnn_features(self.teacher_model.rgb_backbone, x['rgb_left'])
            feature_t_right = self.get_cnn_features(self.teacher_model.rgb_backbone, x['rgb_right'])
            feature_list_t = [feature_t_left, feature_t_front, feature_t_right]
            
            feature_s_front = self.get_cnn_features(self.student_model.rgb_backbone, x['rgb'])
            feature_s_left = self.get_cnn_features(self.student_model.rgb_backbone, x['rgb_left'])
            feature_s_right = self.get_cnn_features(self.student_model.rgb_backbone, x['rgb_right'])
            feature_list_s = [feature_s_left, feature_s_front, feature_s_right]
          
                        # 2. 分层特征提取 
            hierarchical_losses = {}
            views = {'rgb': (feature_s_front, feature_t_front),
                    'rgb_left': (feature_s_left, feature_t_left),
                    'rgb_right': (feature_s_right, feature_t_right)}
                    
            for view_name, (s_feat, t_feat) in views.items():
                # 利用已有中层特征
                s_feats = {
                    'mid': s_feat,  # 重用已提取的特征
                    'early': self.get_hierarchical_features(self.student_model.rgb_backbone, x[view_name])['early'],
                    'late': self.get_hierarchical_features(self.student_model.rgb_backbone, x[view_name])['late']
                }
                t_feats = {
                    'mid': t_feat,  # 重用已提取的特征
                    'early': self.get_hierarchical_features(self.teacher_model.rgb_backbone, x[view_name])['early'],
                    'late': self.get_hierarchical_features(self.teacher_model.rgb_backbone, x[view_name])['late']
                }
                
                for level in ['early', 'mid', 'late']:
                    if level not in hierarchical_losses:
                        hierarchical_losses[level] = 0
                    hierarchical_losses[level] += self.feature_alignment(
                        s_feats[level], t_feats[level], level
                    )
            s_features = self.student_model.forward_features(
                x['rgb'], x['rgb_left'], x['rgb_right'],
                x['rgb_center'], x['lidar'], x['measurements']
            )
            t_features = self.teacher_model.forward_features(
                x['rgb'], x['rgb_left'], x['rgb_right'],
                x['rgb_center'], x['lidar'], x['measurements']
            )
            s_memory = self.student_model.encoder(s_features)
            with torch.no_grad():
                t_memory = self.teacher_model.encoder(t_features)
            
            bs = x['rgb'].shape[0]
            # 学生模型的 tgt 和 query_pos
            tgt_s = self.student_model.position_encoding(
                torch.ones((bs, 1, 20, 20), device=self.device)
            )
            tgt_s = tgt_s.flatten(2)
            tgt_s = torch.cat([tgt_s, self.student_model.query_pos_embed.repeat(bs, 1, 1)], 2)
            tgt_s = tgt_s.permute(2, 0, 1)

            s_hs = self.student_model.decoder(
                self.student_model.query_embed.repeat(1, bs, 1),
                s_memory, query_pos=tgt_s
            )[0]
            # 教师模型的 tgt 和 query_pos
            tgt_t = self.teacher_model.position_encoding(
                torch.ones((bs, 1, 20, 20), device=self.device)
            )
            tgt_t = tgt_t.flatten(2)
            tgt_t = torch.cat([tgt_t, self.teacher_model.query_pos_embed.repeat(bs, 1, 1)], 2)
            tgt_t = tgt_t.permute(2, 0, 1)

            with torch.no_grad():
                t_hs = self.teacher_model.decoder(
                    self.teacher_model.query_embed.repeat(1, bs, 1),
                    t_memory, query_pos=tgt_t
                )[0]
            
            enc_loss, dec_loss = self.enc_dec_alignment(
                s_memory, t_memory,
                s_hs, t_hs
            )

            
            enc_loss, dec_loss = self.enc_dec_alignment(
                s_memory, t_memory,
                s_hs, t_hs
            )

            
        IB_loss = 0
        if self.training:
            IB_loss_t = self.IB.forward([cur_f.detach() for cur_f in  feature_list_t], x)
            IB_loss_s = self.IB.forward([cur_f.detach() for cur_f in  feature_list_s], x)
            IB_loss = IB_loss_t + IB_loss_s

            self.train_IB(IB_loss)
        
            
        
        _, _, z_real = self.IB.forward_encoder(feature_list_t)
        _, _, z_fake = self.IB.forward_encoder(feature_list_s)
        
        z_g_loss = torch.abs(z_real - z_fake).mean()
        
        hierarchical_losses = hierarchical_losses['early']+hierarchical_losses['mid']+hierarchical_losses['late']
        if self.args.loss_kind == 'att':
            if self.base_model == 'interfuser':
                att, entropy, rank_loss = self.wp_attention(x['gt'], x['bev'], x['measurements'], x['actor_pos'], x['actor_cnt'])
                
                return output_s, (wp_g_loss, z_g_loss, att, entropy, rank_loss, IB_loss,hierarchical_losses,enc_loss, dec_loss)
