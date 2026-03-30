import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # 标准的固定位置编码，用sin和cos来编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ContinuousTimeEmbedding(nn.Module):
    """
    连续时间嵌入 (Continuous-Time Embedding) + 时间间隔感知 (Time-Gap Encoding)

    双通道设计：
    1. 绝对时间通道：编码每个时间步的绝对时间位置 t
    2. 时间间隔通道：编码相邻时间步之间的间隔 Δt = t[i] - t[i-1]
       帮助模型理解"距上次观测过了多久"

    输入:
        timestamps: [batch_size, seq_len] 或 [batch_size, seq_len, 1]
    输出:
        time_embed: [batch_size, seq_len, d_model]
    """
    def __init__(self, d_model, learnable_freq=True):
        super(ContinuousTimeEmbedding, self).__init__()
        self.d_model = d_model

        # 绝对时间编码用 d_model//2 维
        # 时间间隔编码用 d_model//2 维
        abs_dim = d_model // 2
        gap_dim = d_model - abs_dim  # 处理 d_model 为奇数的情况

        # --- 绝对时间通道 ---
        half_abs = abs_dim // 2
        if learnable_freq:
            init_freq = (torch.arange(0, half_abs).float() * -(math.log(10000.0) / half_abs)).exp()
            self.abs_freq = nn.Parameter(init_freq, requires_grad=True)
        else:
            freq = (torch.arange(0, half_abs).float() * -(math.log(10000.0) / half_abs)).exp()
            self.register_buffer('abs_freq', freq)
        self.abs_phase = nn.Parameter(torch.zeros(half_abs), requires_grad=True)
        self.abs_linear = nn.Linear(1, abs_dim, bias=True)
        # sin/cos 占 half_abs*2 维, linear 占 abs_dim 维, 融合到 abs_dim
        self.abs_merge = nn.Linear(half_abs * 2 + abs_dim, abs_dim, bias=True)

        # --- 时间间隔通道 ---
        half_gap = gap_dim // 2
        if learnable_freq:
            init_freq_gap = (torch.arange(0, half_gap).float() * -(math.log(1000.0) / half_gap)).exp()
            self.gap_freq = nn.Parameter(init_freq_gap, requires_grad=True)
        else:
            freq_gap = (torch.arange(0, half_gap).float() * -(math.log(1000.0) / half_gap)).exp()
            self.register_buffer('gap_freq', freq_gap)
        self.gap_phase = nn.Parameter(torch.zeros(half_gap), requires_grad=True)
        self.gap_linear = nn.Linear(1, gap_dim, bias=True)
        self.gap_merge = nn.Linear(half_gap * 2 + gap_dim, gap_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        for linear in [self.abs_linear, self.abs_merge, self.gap_linear, self.gap_merge]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, timestamps):
        """
        Args:
            timestamps: [batch_size, seq_len] 或 [batch_size, seq_len, 1]
        Returns:
            [batch_size, seq_len, d_model]
        """
        if timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)  # [B, L, 1]

        # --- 计算时间间隔 Δt ---
        delta_t = torch.zeros_like(timestamps)  # [B, L, 1]
        delta_t[:, 1:, :] = timestamps[:, 1:, :] - timestamps[:, :-1, :]
        # 第一个时间步的 Δt 设为 0

        # --- 绝对时间编码 ---
        abs_angles = timestamps * self.abs_freq.unsqueeze(0).unsqueeze(0) + self.abs_phase.unsqueeze(0).unsqueeze(0)
        abs_periodic = torch.cat([torch.sin(abs_angles), torch.cos(abs_angles)], dim=-1)
        abs_linear = self.abs_linear(timestamps)
        abs_embed = self.abs_merge(torch.cat([abs_periodic, abs_linear], dim=-1))

        # --- 时间间隔编码 ---
        gap_angles = delta_t * self.gap_freq.unsqueeze(0).unsqueeze(0) + self.gap_phase.unsqueeze(0).unsqueeze(0)
        gap_periodic = torch.cat([torch.sin(gap_angles), torch.cos(gap_angles)], dim=-1)
        gap_linear = self.gap_linear(delta_t)
        gap_embed = self.gap_merge(torch.cat([gap_periodic, gap_linear], dim=-1))

        # --- 拼接两个通道 ---
        output = torch.cat([abs_embed, gap_embed], dim=-1)  # [B, L, d_model]

        return output


class Time2Vec(nn.Module):
    """
    Time2Vec 嵌入层 (Kazemi et al., 2019)

    一种经典的连续时间表示方法：
    - 第一个维度是线性函数，捕捉非周期性模式
    - 其余维度是周期性函数(sin)，捕捉周期性模式
    - 所有参数均可学习

    输入:
        timestamps: [batch_size, seq_len] 或 [batch_size, seq_len, 1]
    输出:
        [batch_size, seq_len, d_model]
    """
    def __init__(self, d_model):
        super(Time2Vec, self).__init__()
        self.d_model = d_model

        # 线性分量 (1维)
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))

        # 周期性分量 (d_model - 1 维)
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.randn(d_model - 1))

    def forward(self, timestamps):
        if timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)  # [B, L, 1]

        # 线性分量: [B, L, 1]
        linear = self.w0 * timestamps + self.b0

        # 周期性分量: [B, L, d_model-1]
        periodic = torch.sin(timestamps * self.w.unsqueeze(0).unsqueeze(0)
                             + self.b.unsqueeze(0).unsqueeze(0))

        # 拼接: [B, L, d_model]
        return torch.cat([linear, periodic], dim=-1)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


# ============================================================
# 以下是修改后的 DataEmbedding 类，支持 continuous-time embedding
# ============================================================

class DataEmbedding(nn.Module):
    """
    支持两种模式:
    1. 传统模式 (time_aware=False): value_embedding + position_embedding + temporal_embedding
    2. Time-Aware模式 (time_aware=True): value_embedding + continuous_time_embedding
       此模式下，用连续时间嵌入替换固定位置编码，需要传入实际时间戳
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,
                 time_aware=False, time_embed_type='continuous'):
        super(DataEmbedding, self).__init__()

        self.time_aware = time_aware
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        if time_aware:
            # 使用连续时间嵌入替代固定位置编码
            if time_embed_type == 'time2vec':
                self.time_embedding = Time2Vec(d_model=d_model)
            else:  # 'continuous'
                self.time_embedding = ContinuousTimeEmbedding(d_model=d_model, learnable_freq=True)
        else:
            # 保留原始的位置编码和时间特征嵌入
            self.position_embedding = PositionalEmbedding(d_model=d_model)
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq
            ) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None, mask_token=None, timestamps=None):
        """
        Args:
            x: [B, L, C] 输入序列
            x_mark: [B, L, D] 离散时间特征 (传统模式使用)
            mask_token: [B, L', d_model] mask token (用于预测任务)
            timestamps: [B, L] 或 [B, L, 1] 连续时间戳 (time_aware模式使用)
                        例如: 归一化后的Unix时间戳、小时数等
        """
        if self.time_aware:
            # ---- Time-Aware 模式 ----
            ve = self.value_embedding(x)

            if mask_token is not None:
                ve = torch.cat((ve, mask_token), dim=1)
                # 如果有 mask_token，timestamps 也需要扩展到对应长度
                # 调用方需确保 timestamps 覆盖完整的 total_patch_num

            if timestamps is not None:
                # 确保 timestamps 长度匹配
                t_len = ve.size(1)
                if timestamps.dim() == 2:
                    ts = timestamps[:, :t_len]
                else:
                    ts = timestamps[:, :t_len, :]
                time_embed = self.time_embedding(ts)
                x = ve + time_embed
            else:
                # 如果没提供 timestamps，退化为用等间隔索引作为时间
                batch_size, seq_len = ve.size(0), ve.size(1)
                default_ts = torch.arange(seq_len, dtype=torch.float32, device=ve.device)
                default_ts = default_ts.unsqueeze(0).expand(batch_size, -1)  # [B, L]
                time_embed = self.time_embedding(default_ts)
                x = ve + time_embed
        else:
            # ---- 传统模式 (与原代码完全兼容) ----
            if x_mark is None:
                if mask_token is None:
                    x = self.value_embedding(x) + self.position_embedding(x)
                else:
                    ve = self.value_embedding(x)
                    ve_concat_mask = torch.cat((ve, mask_token), dim=1)
                    x = ve_concat_mask + self.position_embedding(ve_concat_mask)
            else:
                x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)

        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)