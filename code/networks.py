import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class NoisyLinear(nn.Module):
    """噪声线性层 - 用于Noisy Networks探索"""
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 权重和偏置
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        # 初始化参数
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        # 使用独立的高斯噪声
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # 生成噪声
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            return F.linear(x, 
                           self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                           self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DqnNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super(DqnNetwork, self).__init__()
        self.l1 = nn.Linear(inputs, 512)
        self.l2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.leaky_relu(self.l1(x))
        x = self.l2(x)
        return x


class DuelingDqnNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super(DuelingDqnNetwork, self).__init__()
        
        # 共享的特征提取层
        self.feature_layer = nn.Linear(inputs, 512)
        
        # 状态价值头 (V(s))
        self.value_layer = nn.Linear(512, 1)
        
        # 动作优势头 (A(s,a))
        self.advantage_layer = nn.Linear(512, outputs)

    def forward(self, x):
        x = x.to(device)
        
        # 共享特征提取
        features = F.leaky_relu(self.feature_layer(x))
        
        # 计算状态价值
        value = self.value_layer(features)
        
        # 计算动作优势
        advantage = self.advantage_layer(features)
        
        # 使用中心化公式计算Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # 确保在正确的维度上计算平均值
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class PrioritizedNoisyDuelingDQNNetwork(nn.Module):
    """优先噪声Dueling DQN网络架构：
    - Dueling DQN架构：分离状态价值(V)和动作优势(A)估计
    - 支持Noisy Networks：使用可学习的噪声参数进行探索
    - 适度深度特征提取：平衡学习能力和训练稳定性
    - 批量归一化：稳定训练过程
    - ELU激活函数：解决梯度消失问题
    """
    def __init__(self, inputs, outputs, use_noisy_networks=False):
        super(PrioritizedNoisyDuelingDQNNetwork, self).__init__()
        self.use_noisy_networks = use_noisy_networks
        
        # 选择线性层类型
        linear_layer = NoisyLinear if use_noisy_networks else nn.Linear
        
        # 共享的特征提取层 - 适度深度，提高训练稳定性
        self.feature_extractor = nn.Sequential(
            linear_layer(inputs, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            linear_layer(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU()
        )
        
        # 状态价值头 (V(s)) - 独立的隐藏层
        self.value_head = nn.Sequential(
            linear_layer(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            linear_layer(64, 1)
        )
        
        # 动作优势头 (A(s,a)) - 独立的隐藏层
        self.advantage_head = nn.Sequential(
            linear_layer(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            linear_layer(64, outputs)
        )

    def forward(self, x):
        x = x.to(device)
        
        # 确保输入是Float类型
        if x.dtype != torch.float32:
            x = x.float()
        
        # 确保输入是二维的（添加批次维度）
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 深度特征提取
        features = self.feature_extractor(x)
        
        # 计算状态价值
        value = self.value_head(features)
        
        # 计算动作优势
        advantage = self.advantage_head(features)
        
        # 使用中心化公式计算Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        # 如果输入是单个样本，返回一维张量
        if q_values.shape[0] == 1:
            q_values = q_values.squeeze(0)
            
        return q_values
        
    def reset_noise(self):
        """重置所有噪声层的噪声 - 仅用于噪声网络"""
        if self.use_noisy_networks:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


# 为了保持向后兼容性，保留旧的网络类名作为别名
EnhancedPrioritizedNoisyDuelingDqnNetwork = PrioritizedNoisyDuelingDQNNetwork
AdvancedDuelingDqnNetwork = PrioritizedNoisyDuelingDQNNetwork
