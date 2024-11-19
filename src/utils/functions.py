import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num, device=None):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = 1 / (torch.exp(self.log_vars) ** 2)#self.log_vars 是一个参数，表示对应损失函数的对数方差
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]#通过指数函数将 self.log_vars 转换为 precision，然后对每个损失函数乘以精度权重并加上对数方差项，最终得到一个加权求和的损失
            # loss += precision[i] * loss_list[i]
            # print(i,self.log_vars[i])
        return loss #协同方差不确定性（通过这样的设计，可以根据模型的需要来动态地调整各个损失函数的权重，从而更好地进行多任务学习。）

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe
def kl_divergence(p, q):
    """
    计算两个概率分布之间的KL散度。

    参数:
    p -- 概率分布P（期望的分布）
    q -- 概率分布Q（近似的分布）

    返回:
    kl_div -- P和Q之间的KL散度
    """
    # 确保概率分布是有效的概率（即它们的值在0和1之间，并且它们的和为1）
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)

    # 计算KL散度
    # 注意：torch.kl_div计算的是KL散度的负值，所以我们需要取负数
    kl_div = F.kl_div(p.log(), q, reduction='batchmean')  # batchmean用于计算批次中所有样本的平均KL散度

    return kl_div