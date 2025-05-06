import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import time
import sys

sys.path.append("./sasr_net")
sys.path.append("./sasr_net/modules")

from modules.sasr import SourceAwareSemanticRepresentation
from modules.visual_net import resnet18
from modules.qst_net import QstEncoder


class SaSR_Net(nn.Module):
    """
    源感知语义表示网络(Source-aware Semantic Representation Network)
    用于音视频问答任务，能够处理多模态输入(音频、视觉和文本问题)
    """

    def __init__(self, dim_audio: int = 128, dim_visual: int = 512, dim_text: int = 512, dim_inner_embed: int = 512):
        """
        初始化SaSR_Net模型
        
        参数:
            dim_audio: 音频特征的维度
            dim_visual: 视觉特征的维度
            dim_text: 文本特征的维度
            dim_inner_embed: 内部嵌入向量的维度
        """
        super(SaSR_Net, self).__init__()

        self.dim_audio: int = dim_audio
        self.dim_visual: int = dim_visual
        self.dim_test: int = dim_text  # 注意这里有个拼写错误，应该是dim_text
        self.dim_inner_embed: int = dim_inner_embed

        # 音频特征处理模块
        self.fc_a1 = nn.Linear(dim_audio, dim_inner_embed)  # 音频特征第一层映射
        self.fc_a2 = nn.Linear(dim_inner_embed, dim_inner_embed)  # 音频特征第二层映射

        # 视觉特征提取器 - 使用预训练的ResNet18
        self.visual_net = resnet18(pretrained=True)

        # 特征融合模块
        self.fc_fusion = nn.Linear(dim_inner_embed * 2, dim_inner_embed)  # 融合两种模态的特征

        # 自注意力机制相关层 - 层1
        self.linear11 = nn.Linear(dim_inner_embed, dim_inner_embed)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(dim_inner_embed, dim_inner_embed)

        # 自注意力机制相关层 - 层2
        self.linear21 = nn.Linear(dim_inner_embed, dim_inner_embed)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(dim_inner_embed, dim_inner_embed)
        self.norm1 = nn.LayerNorm(dim_inner_embed)
        self.norm2 = nn.LayerNorm(dim_inner_embed)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(dim_inner_embed)

        # 多头注意力机制 - 分别用于音频和视觉特征
        self.attn_a = nn.MultiheadAttention(dim_inner_embed, 4, dropout=0.1)  # 音频特征的多头注意力
        self.attn_v = nn.MultiheadAttention(dim_inner_embed, 4, dropout=0.1)  # 视觉特征的多头注意力

        # 问题编码器 - 处理文本问题
        self.question_encoder = QstEncoder(
            93, dim_inner_embed, dim_inner_embed, 1, dim_inner_embed)  # 93为词汇表大小

        # 答案生成相关层
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc_ans = nn.Linear(dim_inner_embed, 42)  # 42为答案类别数

        # 全局特征池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl = nn.Linear(dim_inner_embed * 2, dim_inner_embed)

        # 特征融合和分类层
        self.fc1 = nn.Linear(dim_inner_embed * 2, dim_inner_embed)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(dim_inner_embed, dim_inner_embed // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(dim_inner_embed // 2, dim_inner_embed // 4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(dim_inner_embed // 4, 2)  # 二分类输出
        self.relu4 = nn.ReLU()

        # 源感知语义表示模块
        self.sasr = SourceAwareSemanticRepresentation()

    def forward(self, audio, visual_posi, visual_nega, question):
        '''
        模型前向传播函数
        
        参数:
            question: 问题特征 [B, T]，B为批次大小，T为序列长度
            audio: 音频特征 [B, T, C]，C为通道数
            visual_posi: 正样本视觉特征 [B, T, C, H, W]，H和W为高和宽
            visual_nega: 负样本视觉特征 [B, T, C, H, W]
            
        返回:
            out_qa: 问答输出
            out_match_posi: 正样本匹配分数
            out_match_nega: 负样本匹配分数
            av_cls_prob_posi: 正样本音视频分类概率
            v_prob_posi: 正样本视觉概率
            a_prob_posi: 正样本音频概率
            mask_posi: 注意力掩码
        '''

        # 打印输入张量的形状
        print(f"Input shapes: audio={audio.shape}, visual_posi={visual_posi.shape}, visual_nega={visual_nega.shape}, question={question.shape}")

        # 处理问题特征
        qst_feature = self.question_encoder(question)  # 编码问题
        xq = qst_feature.unsqueeze(0)  # 添加维度用于注意力机制

        ###############################################################################################
        # 处理正样本视频

        audio_feat_posi, visual_feat_grd_posi, out_match_posi, av_cls_prob_posi, v_prob_posi, a_prob_posi, mask_posi = self.out_match_infer(
            audio, visual_posi)  # 处理正样本

        ###############################################################################################

        ###############################################################################################
        # 处理负样本视频

        audio_feat_nega, visual_feat_grd_nega, out_match_nega, av_cls_prob_nega, v_prob_nega, a_prob_nega, mask_posi = self.out_match_infer(
            audio, visual_nega)  # 处理负样本

        ###############################################################################################

        B = xq.shape[1]  # 批次大小
        visual_feat_grd_be = visual_feat_grd_posi.view(
            B, -1, self.dim_inner_embed)   # [B, T, 512] 重塑视觉特征
        visual_feat_grd = visual_feat_grd_be.permute(1, 0, 2)  # 调整维度顺序为[T, B, C]

        # 打印中间特征的形状
        print(f"Intermediate shapes: visual_feat_grd={visual_feat_grd.shape}")

        # 问题对视觉特征的注意力机制
        visual_feat_att = self.attn_v(
            xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear12(self.dropout1(
            F.relu(self.linear11(visual_feat_att))))  # 前馈网络
        visual_feat_att = visual_feat_att + self.dropout2(src)  # 残差连接
        visual_feat_att = self.norm1(visual_feat_att)  # 层归一化

        # 问题对音频特征的注意力机制
        audio_feat_be = audio_feat_posi.view(B, -1, self.dim_inner_embed)
        audio_feat = audio_feat_be.permute(1, 0, 2)  # 调整维度顺序为[T, B, C]
        audio_feat_att = self.attn_a(
            xq, audio_feat, audio_feat, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(
            F.relu(self.linear21(audio_feat_att))))  # 前馈网络
        audio_feat_att = audio_feat_att + self.dropout4(src)  # 残差连接
        audio_feat_att = self.norm2(audio_feat_att)  # 层归一化

        # 融合音频和视觉特征
        feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(),
                         visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)  # 非线性激活
        feat = self.fc_fusion(feat)  # 特征融合

        # 与问题特征融合
        combined_feature = torch.mul(feat, qst_feature)  # 元素乘法，实现特征交互
        combined_feature = self.tanh(combined_feature)  # 非线性激活
        
        # 生成答案预测
        out_qa = self.fc_ans(combined_feature)  # [batch_size, ans_vocab_size]

        # 打印最终输出的形状
        print(f"Output shapes: out_qa={out_qa.shape}, out_match_posi={out_match_posi.shape}, out_match_nega={out_match_nega.shape}")

        return out_qa, out_match_posi, out_match_nega, av_cls_prob_posi, v_prob_posi, a_prob_posi, mask_posi

    def out_match_infer(self, audio, visual):
        """
        音视频匹配推断函数
        
        参数:
            audio: 音频特征
            visual: 视觉特征
            
        返回:
            grouped_audio_embedding: 分组后的音频嵌入
            visual_feat_grd: 视觉特征定位后的表示
            out_match: 匹配得分
            av_cls_prob: 音视频分类概率
            v_prob: 视觉概率
            a_prob: 音频概率
            x2_p: 注意力掩码
        """

        # 处理音频特征 [2*B*T, 128]
        audio_feat = F.relu(self.fc_a1(audio))  # 第一层特征提取
        audio_feat = self.fc_a2(audio_feat)  # 第二层特征提取
        audio_feat_pure = audio_feat  # 保存纯音频特征
        B, T, C = audio_feat.size()  # [B, T, C]
        audio_feat = audio_feat.view(B, T, C)  # 重塑为[B*T, C]

        # 处理视觉特征 [2*B*T, C, H, W]
        B, T, C, H, W = visual.size()
        temp_visual = visual.view(B*T, C, H, W)  # 重塑为[B*T, C, H, W]
        # 全局池化得到视觉特征 [B*T, C, 1, 1]
        v_feat = self.avgpool(temp_visual)
        visual_feat_before_grounding = v_feat.squeeze()  # [B*T, C]
        visual_feat_before_grounding = visual_feat_before_grounding.view(
            B, -1, C)  # 重塑为[B, T, C]

        # 调用SASR模块处理特征
        _, _, av_cls_prob, a_prob, v_prob, grouped_audio_embedding, grouped_visual_embedding = self.sasr(
            audio_feat, visual_feat_before_grounding, visual_feat_before_grounding)

        (B, C, H, W) = temp_visual.size()
        # 重塑视觉特征为 [B*T, C, HxW]
        v_feat = temp_visual.view(B, C, H * W)
        # 调整维度顺序为 [B, HxW, C]
        v_feat = v_feat.permute(0, 2, 1)
        visual_feat = nn.functional.normalize(v_feat, dim=2)  # 特征归一化 [B, HxW, C]

        # 音视频定位 - 计算音频和视觉特征的相关性
        (B, T, C) = grouped_audio_embedding.size()
        audio_feat_aa = grouped_audio_embedding.view(
            B * T, -1).unsqueeze(dim=-1)  # [B*T, C, 1]
        audio_feat_aa = nn.functional.normalize(
            audio_feat_aa, dim=1)  # 特征归一化 [B*T, C, 1]

        # 计算视觉特征与音频特征的注意力权重
        x2_va = torch.matmul(
            visual_feat, audio_feat_aa).squeeze()  # [B*T, HxW]

        # 生成注意力掩码 [B*T, 1, HxW]
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)
        # 通过注意力机制对视觉特征进行加权
        visual_feat_grd = torch.matmul(x2_p, visual_feat)
        # [B*T, C]
        visual_feat_grd_after_grounding = visual_feat_grd.squeeze()

        # 融合全局和定位后的视觉特征
        grouped_visual_embedding = grouped_visual_embedding.flatten(0, 1)
        visual_gl = torch.cat(
            (grouped_visual_embedding, visual_feat_grd_after_grounding), dim=-1)
        visual_feat_grd = self.tanh(visual_gl)  # 非线性激活
        visual_feat_grd = self.fc_gl(visual_feat_grd)  # 特征融合 [B*T, C]

        grouped_audio_embedding = grouped_audio_embedding.flatten(0, 1)

        # 融合音频和视觉特征 [B*T, C*2]
        feat = torch.cat((grouped_audio_embedding, visual_feat_grd), dim=-1)

        # 多层前馈网络处理融合特征
        feat = F.relu(self.fc1(feat))  # (1024, 512)
        feat = F.relu(self.fc2(feat))  # (512, 256)
        feat = F.relu(self.fc3(feat))  # (256, 128)
        out_match = self.fc4(feat)  # (128, 2) 输出匹配得分

        return grouped_audio_embedding, visual_feat_grd, out_match, av_cls_prob, v_prob, a_prob, x2_p
