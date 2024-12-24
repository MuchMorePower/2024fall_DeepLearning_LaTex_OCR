import numpy as np
import torch
import math
from torch import nn
import torchvision
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=False)
        # 替换初始卷积层
        resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        resnet.maxpool = nn.Identity()  # 去掉最大池化层
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的全连接层和平均池化层


    def forward(self, x):
        x = self.encoder(x)

        #位置嵌入
        x = x.permute(0,2,3,1)
        x = self.add_timing_signal_nd(x)
        #x = x.permute(0,3,1,2)

        x = x.contiguous()

        return x

    def add_timing_signal_nd(self, x, min_timescale=1.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.
        Args:
            x: a Tensor with shape [batch, d1 ... dn, channels]
            min_timescale: a float
            max_timescale: a float

        Returns:
            a Tensor the same shape as x.

        """
        static_shape = list(x.shape) # [2, 512, 50, 120]
        num_dims = len(static_shape) - 2  # 2
        channels = x.shape[-1]  # 512 
        num_timescales = channels // (num_dims * 2)  # 512 // (2*2) = 128
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
            torch.FloatTensor([i for i in range(num_timescales)]) * -log_timescale_increment)  # len == 128
        for dim in range(num_dims):  # dim == 0; 1
            length = x.shape[dim + 1]  # 要跳过前两个维度
            position = torch.arange(length).float()  # len == 50
            scaled_time = torch.reshape(position,(-1,1)) * torch.reshape(inv_timescales,(1,-1))
            #[50,1] x [1,128] = [50,128]
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1).to(device)  # [50, 256]
            prepad = dim * 2 * num_timescales  # 0; 256
            postpad = channels - (dim + 1) * 2 * num_timescales  # 512-(1;2)*2*128 = 256; 0
            signal = F.pad(signal, (prepad,postpad,0,0))  # [50, 512]
            for _ in range(1 + dim):  # 1; 2
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):  # 1, 0
                signal = signal.unsqueeze(-2)
            # don't use +=, or the in-place calculation will raise error in backward
            x = x + signal  # [1, 14, 1, 512]; [1, 1, 14, 512]
        return x

class Attention(nn.Module):
    """
    自适应注意力网络。
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        初始化注意力模块。

        :param encoder_dim: 编码器输出的特征维度 (512)
        :param decoder_dim: 解码器 RNN 的隐藏状态维度 (512)
        :param attention_dim: 注意力网络的中间维度 (256)
        :param dropout_p: Dropout概率 (0.1)
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)    # 将编码器输出映射到注意力空间
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)    # 将解码器隐藏状态映射到注意力空间
        self.full_att = nn.Linear(attention_dim, 1)                 # 生成注意力得分

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)                            # 计算注意力权重
        self.dropout = nn.Dropout(p=0.1)                            # Dropout层

        # 添加自适应门控机制
        self.attention_gate = nn.Sequential(
            nn.Linear(decoder_dim, attention_dim),  # 转换解码器隐藏状态
            nn.ReLU(),
            nn.Linear(attention_dim, 1)             # 生成缩放因子
        )

    def forward(self, encoder_out, decoder_hidden):
        """
        前向传播。

        :param encoder_out: 编码器输出，形状为 (batch_size, num_pixels, encoder_dim=512)
        :param decoder_hidden: 解码器当前隐藏状态，形状为 (batch_size, decoder_dim=512)
        :return: 加权编码器输出 (batch_size, encoder_dim=512)，注意力权重 (batch_size, num_pixels)
        """
        # 计算注意力得分
        att1 = self.encoder_att(encoder_out)                        # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)                     # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)

        # 应用Dropout
        att = self.dropout(att)  # (batch_size, num_pixels)

        # 计算注意力权重
        alpha = self.softmax(att)                                   # (batch_size, num_pixels)

        # 计算注意力加权的编码器输出
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        # 应用自适应门控机制，动态缩放注意力加权的编码器输出
        attention_scale = torch.sigmoid(self.attention_gate(decoder_hidden))  # (batch_size, 1)
        attention_weighted_encoding = attention_weighted_encoding * attention_scale  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    """
    带有自适应注意力机制和双层 GRU 的解码器，包含残差连接和层归一化，最后加上前馈神经网络。
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=512, dropout=0.1, p=1.0):
        """
        初始化解码器。

        :param attention_dim: 注意力网络的中间维度 (256)
        :param embed_dim: 词嵌入的维度 (256)
        :param decoder_dim: 解码器 GRU 的隐藏状态维度 (512)
        :param vocab_size: 词汇表的大小
        :param encoder_dim: 编码器输出的特征维度 (512)
        :param dropout: dropout 概率
        :param p: teacher forcing 的概率
        :param ffn_dim: 前馈神经网络的隐藏维度 (1024)
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.p = p  # teacher forcing 的概率
        self.ffn_dim = 1024  # 前馈神经网络的隐藏层维度

        # 注意力网络
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        
        # 双层 GRU 单元
        self.gru1 = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.gru2 = nn.GRUCell(decoder_dim, decoder_dim)

        # 投影层，将第一层 GRU 的输入投影到 decoder_dim
        self.input_projection = nn.Linear(embed_dim + encoder_dim, decoder_dim)

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(decoder_dim)
        self.layer_norm2 = nn.LayerNorm(decoder_dim)

        # 前馈神经网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(self.ffn_dim, vocab_size)  # 输出层映射到 vocab_size
        )

        self.init_weights()  # 初始化权重

    def init_weights(self):
        """
        用均匀分布初始化部分参数，以便更快收敛。
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        # 为 FFN 的全连接层进行 Kaiming 初始化
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                # Kaiming 正态初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)  # 偏置初始化为0

    def init_hidden_state(self, batch_size):
        """
        初始化解码器中双层 GRU 的隐藏状态。

        :param batch_size: 批量大小
        :return: 初始化的隐藏状态 h1 和 h2
        """
        h1 = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        h2 = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        return h1, h2

    def forward(self, encoder_out, encoded_captions=None, caption_lengths=None, p=1.0):
        self.p = p

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # 展平成 (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        sort_ind = None

        if encoded_captions is None or caption_lengths is None:
            max_caption_length = 16
            encoded_captions = torch.zeros(batch_size, max_caption_length, dtype=torch.long).to(encoder_out.device)
            caption_lengths = torch.full((batch_size,), max_caption_length, dtype=torch.long).to(encoder_out.device)
        else:
            caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
            encoder_out = encoder_out[sort_ind]
            encoded_captions = encoded_captions[sort_ind]
            caption_lengths = caption_lengths.long()

        embeddings = self.embedding(encoded_captions)

        # 初始化 GRU 隐藏状态
        h1, h2 = self.init_hidden_state(batch_size)

        decode_lengths = caption_lengths - 1
        predictions = torch.zeros(batch_size, int(max(decode_lengths)), vocab_size).to(device)
        alphas = torch.zeros(batch_size, int(max(decode_lengths)), num_pixels).to(device)

        for t in range(int(max(decode_lengths))):
            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h2[:batch_size_t]
            )

            if t == 0 or (torch.rand(1).item() < self.p):
                input_embedding = embeddings[:batch_size_t, t, :]
            else:
                prev_preds = predictions[:batch_size_t, t - 1, :]
                prev_words = prev_preds.argmax(dim=1)
                input_embedding = self.embedding(prev_words)

            gru1_input = torch.cat([input_embedding, attention_weighted_encoding], dim=1)

            h1_prev = h1[:batch_size_t]
            h1_new = self.gru1(gru1_input, h1_prev)

            input_proj = self.input_projection(gru1_input)

            h1_res = h1_new + input_proj.clone()
            h1_res = self.layer_norm1(h1_res.clone())

            h2_prev = h2[:batch_size_t]
            h2_new = self.gru2(h1_res, h2_prev)

            # 先对 GRU 输出 h2_new 应用 dropout，再进行残差连接
            h2_res = self.dropout_layer(h2_new)  # 先应用 dropout
            h2_res = h2_res + h1_res.clone()  # 然后进行残差连接
            h2_res = self.layer_norm2(h2_res.clone())

            # 通过前馈神经网络处理 h2_res
            preds = self.ffn(h2_res)  # 使用前馈神经网络进行词汇映射
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            # 深拷贝更新隐藏状态
            h1_updated = h1.clone()
            h2_updated = h2.clone()

            h1_updated[:batch_size_t] = h1_new
            h2_updated[:batch_size_t] = h2_new

            h1 = h1_updated
            h2 = h2_updated

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

