from torch import nn
import torch
from modules.encoders import LanguageEmbeddingLayer, SubNet
from src.utils.functions import kl_divergence
from utils.Tri_aug_attention import Tri_aug_attlayer,F_S_Decoder
from utils.TimeEncoder import TimeEncoder,ConvLayer

class MSA(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Args:
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super(MSA,self).__init__()
        self.hp = hp
        self.text_enc = LanguageEmbeddingLayer(hp)
        # Trimodal Settings

        self.act_t = SubNet(
            in_size= 2*hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )

        self.fusion_prj_s = SubNet(
            in_size= 3*hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fusion_prj_f =SubNet(
            in_size= hp.d_prjh,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )
        self.fc1=nn.Linear(768,hp.d_prjh)
        self.a_shape = nn.AdaptiveMaxPool1d(hp.a_size)
        self.v_shape = nn.AdaptiveMaxPool1d(hp.v_size)
        self.t_shrink = ConvLayer(hp.d_prjh, dis_len=hp.dislen)
        self.a_shrink = ConvLayer(hp.d_prjh, dis_len=hp.dislen)
        self.v_shrink = ConvLayer(hp.d_prjh, dis_len=hp.dislen)
        # self.t_shrink = nn.AdaptiveMaxPool1d(hp.dislen)
        # self.a_shrink = nn.AdaptiveMaxPool1d(hp.dislen)
        # self.v_shrink = nn.AdaptiveMaxPool1d(hp.dislen)

        self.ET_TA = Tri_aug_attlayer(hp)
        self.ET_TV = Tri_aug_attlayer(hp)
        # self.ET_AV = Tri_aug_attlayer(hp)
        self.decoder_t = F_S_Decoder(hp)
        self.decoder_a = F_S_Decoder(hp)
        self.decoder_v = F_S_Decoder(hp)

        # self.sinkhorn = SinkhornDistance()
        self.sentiment_fc1 = nn.Sequential(
            nn.Linear(hp.d_prjh, hp.d_prjh),
            nn.Tanh(),
            nn.Linear(hp.d_prjh, hp.d_prjh)
        )
        self.a_encoder = TimeEncoder(self.hp,encoder_layers=hp.tse_layers,step_ratio=hp.step_ratio, data_type='a')
        self.v_encoder = TimeEncoder(self.hp,encoder_layers=hp.tse_layers,step_ratio=hp.step_ratio, data_type='v')
    def forward(self,visual, acoustic, v_len, a_len, bert_sent,  bert_sent_mask):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(bert_sent,bert_sent_mask) # (batch_size, seq_len, emb_size)
        lang_emb=self.fc1(enc_word)

        acoustic = self.a_shape(acoustic.permute(0, 2, 1)).permute(0, 2, 1)
        visual = self.v_shape(visual.permute(0, 2, 1)).permute(0, 2, 1)
        aco_emb=self.a_encoder(acoustic)
        vis_emb=self.v_encoder(visual)

        lang_emb = self.t_shrink(lang_emb)
        aco_emb = self.a_shrink(aco_emb)
        vis_emb = self.v_shrink(vis_emb)
        S_t =torch.mean(lang_emb, dim=1)
        S_a =torch.mean(aco_emb, dim=1)
        S_v =torch.mean(vis_emb, dim=1)
        S_feat = torch.cat((S_t,S_a,S_v), dim=-1)
        _,r_preds = self.fusion_prj_s(S_feat)#通过单模态标签损失促使TSE捕获模态特定的情感信息
        TF_v, VF_t, TM, IM = self.ET_TV(lang_emb, vis_emb)
        TF_a, AF_t, TM, IM = self.ET_TA(lang_emb, aco_emb)
        # AF_v, VF_a, TM, IM = self.ET_AV(aco_emb, vis_emb)
        t_emb = torch.cat((TF_a, TF_v), dim=-1)
        t_emb,_ = self.act_t(t_emb)

        T_dfeat = torch.mean(t_emb, dim=1)
        A_dfeat = torch.mean(AF_t, dim=1)
        V_dfeat = torch.mean(VF_t, dim=1)
        # F_feat,r_preds_F = self.fusion_prj4(F_feat)

        # 将特征投影到情感空间
        T_emb = self.sentiment_fc1(T_dfeat)
        A_emb = self.sentiment_fc1(A_dfeat)
        V_emb = self.sentiment_fc1(V_dfeat)

        # 使用欧氏距离计算距离
        cost1 = torch.norm(T_emb - A_emb, dim=-1, p=2)
        cost2 = torch.norm(T_emb - V_emb, dim=-1, p=2)
        cost3 = torch.norm(A_emb - V_emb, dim=-1, p=2)
        # 扩展距离以匹配特征维度
        cost_ta = cost1.unsqueeze(-1)
        cost_tv = cost2.unsqueeze(-1)
        cost_av = cost3.unsqueeze(-1)

        # 计算距离的倒数，避免除以零
        epsilon = 1e-6  # 避免除以零的小正则项
        inv_T = 1 / (cost_ta + cost_tv + epsilon)
        inv_A = 1 / (cost_ta + cost_av + epsilon)
        inv_V = 1 / (cost_tv + cost_av + epsilon)

        # 归一化倒数以得到权重
        total_inv_cost = inv_T + inv_A + inv_V
        weight_T = inv_T / total_inv_cost
        weight_A = inv_A / total_inv_cost
        weight_V = inv_V / total_inv_cost

        # 特征融合
        F_feat = weight_T * T_dfeat + weight_A * A_dfeat + weight_V * V_dfeat
        # F_feat =T_dfeat+A_dfeat+V_dfeat
        F_feat,r_preds_F = self.fusion_prj_f(F_feat)#通过融合预测损失捕获多模态交互情感信息

        # F_feat = torch.cat((lang_emb, aco_emb, vis_emb), dim=1)
        # F_feat = self.f_shrink(F_feat)
        t_recon = self.decoder_t(t_emb,lang_emb).mean(dim=1)
        a_recon = self.decoder_a(AF_t,aco_emb).mean(dim=1)
        v_recon = self.decoder_v(VF_t,vis_emb).mean(dim=1)
        kl_div_t_recon = kl_divergence(S_t, t_recon)
        kl_div_a_recon = kl_divergence(S_a, a_recon)
        kl_div_v_recon = kl_divergence(S_v, v_recon)
        kl_div_t_distill = kl_divergence(F_feat, S_t)
        kl_div_a_distill = kl_divergence(F_feat, S_a)
        kl_div_v_distill = kl_divergence(F_feat, S_v)#双向kl散度
        L_recon = (kl_div_t_recon + kl_div_a_recon + kl_div_v_recon) / 3
        L_dis = (kl_div_t_distill + kl_div_a_distill + kl_div_v_distill) / 3

        return r_preds,r_preds_F,L_recon,L_dis
