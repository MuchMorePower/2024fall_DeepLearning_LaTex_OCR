import numpy as np
import torch
import math
from torch import nn
import torchvision
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        # 定义共享的 MLP（全连接层）
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # 进行全局平均池化和最大池化
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)  # (batch_size, channels, 1, 1)
        max_pool = torch.max(x, dim=2, keepdim=True)[0]  # (batch_size, channels, 1, width)
        max_pool = torch.max(max_pool, dim=3, keepdim=True)[0]  # (batch_size, channels, 1, 1)

        # 使用共享 MLP 处理平均池化和最大池化的结果
        avg_out = self.shared_mlp(avg_pool.view(batch_size, channels))  # (batch_size, channels)
        avg_out = avg_out.view(batch_size, channels, 1, 1)  # (batch_size, channels, 1, 1)

        max_out = self.shared_mlp(max_pool.view(batch_size, channels))  # (batch_size, channels)
        max_out = max_out.view(batch_size, channels, 1, 1)  # (batch_size, channels, 1, 1)

        # 相加得到最终的通道注意力
        attention = avg_out + max_out  # (batch_size, channels, 1, 1)

        # 将输入特征图和注意力相乘，得到加权后的特征图
        return x * attention  # (batch_size, channels, height, width)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        # 拼接平均池化和最大池化的结果
        spatial_attention_input = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, 2, height, width)
        # 生成空间注意力图
        attention_map = self.conv(spatial_attention_input)  # (batch_size, 1, height, width)
        attention_map = self.sigmoid(attention_map)  # (batch_size, 1, height, width)
        
        return x * attention_map  # (batch_size, channels, height, width)

class CBAM(nn.Module):
    def __init__(self, in_channels, kernel_size=7, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力
        x = self.channel_attention(x) 
        # 然后应用空间注意力
        x = self.spatial_attention(x) 


        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
        )
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.channel_attention = ChannelAttention(growth_rate)

    def forward(self, x):
        # 手动逐层执行 bottleneck 中的操作
        x = self.bottleneck[0](x)  # Conv2d
        x = self.bottleneck[1](x)  # BatchNorm2d
        x = self.bottleneck[2](x)  # ReLU
        x = self.dropout1(x)
        x = self.bottleneck[3](x)  # Conv2d
        x = self.bottleneck[4](x)  # BatchNorm2d
        x = self.bottleneck[5](x)  # ReLU
        x = self.dropout2(x)
        
        # 执行通道注意力
        x = self.channel_attention(x)

        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, use_bottleneck=False):
        super(DenseBlock, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if self.use_bottleneck:
                self.layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.BatchNorm2d(in_channels + i * growth_rate),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
                    )
                )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.AvgPool2d(kernel_size=2, stride=2)  # 下采样
        )
    
    def forward(self, x):
        return self.transition(x)


class CBAMBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=32):
        super(CBAMBasicBlock, self).__init__()
        # 第一层卷积，使用分组卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二层卷积，使用分组卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # downsample用于处理输入输出通道不一致的情况
        self.downsample = downsample
        
        # CBAM模块
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        identity = x
        
        # 第一层卷积
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 第二层卷积
        x = self.bn2(self.conv2(x))
        
        # 如果需要降采样，应用降采样
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # CBAM模块
        x = self.cbam(x)
        
        # 将卷积输出与identity（可能经过downsample）相加
        x += identity
        x = self.relu(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self,growth_rate=32,num_layers=16):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=False)
        
        

        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.downSample = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1, bias=False),  # 下采样
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

        
        # 提取 ResNet 的各层
        self.layer0 = self._make_layer(32, 32, 2)
        self.layer1 = self._make_layer(64, 128, 6)  # 第一个残差块，使用CBAMBasicBlock
        self.layer2 = self._make_layer(128, 256, 6, stride=2)  # 第二个残差块
        # 使用 DenseNet 的后两层
        self.dense_block1 = DenseBlock(256, growth_rate, num_layers, use_bottleneck=True)  # DenseNet 第一层

        num_channels = 256 + growth_rate * num_layers  # 更新通道数
        self.channel_attention1 = ChannelAttention(num_channels)

        self.transition1 = TransitionLayer(num_channels, num_channels//2)  # TransitionLayer
        num_channels = num_channels//2
        self.dense_block2 = DenseBlock(num_channels, growth_rate, num_layers, use_bottleneck=True)  # DenseNet 第二层
        num_channels += growth_rate * num_layers
        self.channel_attention2 = ChannelAttention(num_channels)

        #print(num_channels)
        self.final_conv = nn.Conv2d(num_channels,512,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn = nn.BatchNorm2d(512)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers.append(CBAMBasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(CBAMBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layer0(x)
        x = self.downSample(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dense_block1(x)
        x = self.channel_attention1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)
        x = self.channel_attention2(x)
        x = self.final_conv(x)
        x = self.bn(x)
        
        # 将输出维度从 (batch, channel, height, width) 转为 (batch, height, width, channel)
        x = x.permute(0, 2, 3, 1)  # 调整输出的维度顺序（batch, height, width, channels）
        x = x.contiguous()

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
    带有自适应注意力机制和三层 GRU 的解码器，包含残差连接和层归一化。
    """
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=512, dropout=0.5, p=1.0):
        """
        初始化解码器。

        :param attention_dim: 注意力网络的中间维度 (256)
        :param embed_dim: 词嵌入的维度 (256)
        :param decoder_dim: 解码器 GRU 的隐藏状态维度 (512)
        :param vocab_size: 词汇表的大小
        :param encoder_dim: 编码器输出的特征维度 (512)
        :param dropout: dropout 概率
        :param p: teacher forcing 的概率
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.p = p  # teacher forcing 的概率

        # 注意力网络
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # 双层 GRU 单元
        self.gru1 = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        self.gru2 = nn.GRUCell(decoder_dim, decoder_dim)
        self.gru3 = nn.GRUCell(decoder_dim, decoder_dim)

        # 投影层，将第一层 GRU 的输入投影到 decoder_dim
        self.input_projection = nn.Linear(embed_dim + encoder_dim, decoder_dim)

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(decoder_dim)
        self.layer_norm2 = nn.LayerNorm(decoder_dim)
        self.layer_norm3 = nn.LayerNorm(decoder_dim)

        
        # 输出层，将解码器的输出映射到词汇表大小
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()  # 初始化权重

    def init_weights(self):
        """
        用均匀分布初始化部分参数，以便更快收敛。
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        

    def init_hidden_state(self, batch_size):
        """
        初始化解码器中双层 GRU 的隐藏状态。

        :param batch_size: 批量大小
        :return: 初始化的隐藏状态 h1 和 h2
        """
        h1 = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        h2 = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        h3 = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        return h1, h2, h3

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
        h1, h2, h3 = self.init_hidden_state(batch_size)

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

            h2_res = h2_new + h1_res.clone()
            h2_res = self.layer_norm2(h2_res.clone())

            # 第三层 GRU
            h3_prev = h3[:batch_size_t]
            h3_new = self.gru3(h2_res, h3_prev)

            h3_res = h3_new + h2_res.clone()
            h3_res = self.layer_norm3(h3_res.clone())


            preds = self.fc(self.dropout_layer((h3_res)))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            # 深拷贝更新隐藏状态
            h1_updated = h1.clone()
            h2_updated = h2.clone()
            h3_updated = h3.clone()

            h1_updated[:batch_size_t] = h1_new
            h2_updated[:batch_size_t] = h2_new
            h3_updated[:batch_size_t] = h3_new

            h1 = h1_updated
            h2 = h2_updated
            h3 = h3_updated
            
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    #predictions 形状: (batch_size, max_decode_length, vocab_size)
    #encoded_captions 形状: (batch_size, max_caption_length)
    #decode_lengths 形状: (batch_size)
    #alphas 形状: (batch_size, max_decode_length, num_pixels)
    #sort_ind 形状: (batch_size)
    # 如果对输入的 caption_lengths 进行了排序（因为 caption_lengths.sort() 会排序样本以处理不同长度的序列），那么 sort_ind 将包含排序的索引，用于恢复解码后的顺序
    def beam_search_v0(self, encoder_out, word_map, beam_size=3, max_caption_length=50):
        """
        使用束搜索进行序列生成。

        :param encoder_out: 编码器输出，形状为 (batch_size, num_pixels, encoder_dim)
        :param word_map: 词汇表映射（词到索引）
        :param beam_size: 束宽，表示同时保留的候选序列数量
        :param max_caption_length: 生成的最大序列长度
        :return: 最优序列及其得分
        """
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "Beam Search currently only supports batch size = 1."

        # 初始化解码器的隐藏状态
        h1, h2, h3 = self.init_hidden_state(batch_size)  # (1, decoder_dim)

        # 展平成 (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(1, -1, self.encoder_dim)
        num_pixels = encoder_out.size(1)

        # 初始化束搜索的序列和得分
        k = beam_size
        sequences = [[word_map["<start>"]]]  # 初始序列，开始符号
        scores = torch.zeros(k).to(device)  # 初始得分
        hidden_states = [(h1.clone(), h2.clone(), h3.clone()) for _ in range(k)]  # 隐状态副本

        """hidden_states = [
            (h1_1, h2_1, h3_1),  # 第 1 个束
            (h1_2, h2_2, h3_2),  # 第 2 个束
            ...
            (h1_k, h2_k, h3_k)   # 第 k 个束
            ]
        """

        # 缓存注意力权重，用于调试或解释（可选）
        alphas = []

        # 对每个时间步进行解码
        for t in range(max_caption_length):
            all_candidates = []  # 当前时间步的所有候选序列

            # 对当前的束中的每个序列进行扩展
            for i, seq in enumerate(sequences):
                # 如果序列已包含结束符，则直接加入候选
                if seq[-1] == word_map["<end>"]:
                    all_candidates.append((seq, scores[i], hidden_states[i]))
                    continue

                # 获取当前序列的最后一个词作为输入
                input_word = torch.LongTensor([seq[-1]]).to(device)  # (1)
                input_embedding = self.embedding(input_word)  # (1, embed_dim)

                # 计算注意力
                attention_weighted_encoding, alpha = self.attention(
                    encoder_out, hidden_states[i][1]  # 使用 h2 作为注意力的查询
                )  # (1, encoder_dim), (1, num_pixels)

                # 拼接嵌入和注意力向量，保持批量维度
                gru1_input = torch.cat([input_embedding, attention_weighted_encoding], dim=1)

                h1_prev = hidden_states[i][0]
                h1_new = self.gru1(gru1_input, h1_prev)

                input_proj = self.input_projection(gru1_input)

                h1_res = h1_new + input_proj.clone()
                h1_res = self.layer_norm1(h1_res.clone())

                h2_prev = hidden_states[i][1]
                h2_new = self.gru2(h1_res, h2_prev)

                h2_res = h2_new + h1_res.clone()
                h2_res = self.layer_norm2(h2_res.clone())

                # 第三层 GRU
                h3_prev = hidden_states[i][2]
                h3_new = self.gru3(h2_res, h3_prev)

                h3_res = h3_new + h2_res.clone()
                h3_res = self.layer_norm3(h3_res.clone())

                # 预测下一个词
                scores_logits = self.fc(h3_res)  # (1, vocab_size)
                scores_logits = torch.log_softmax(scores_logits, dim=1)  # (1, vocab_size)

                # 获取 top beam_size 个候选词
                top_log_probs, top_indices = scores_logits.topk(beam_size)  # (1, beam_size)

                # 扩展当前束的每个候选词
                for j in range(beam_size):
                    next_token = top_indices[0, j].item()
                    next_log_prob = top_log_probs[0, j].item()
                    new_seq = seq + [next_token]
                    new_score = scores[i] + next_log_prob
                    new_hidden_states = (h1_new.clone(), h2_new.clone(), h3_new.clone())
                    all_candidates.append((new_seq, new_score, new_hidden_states))

                # 保存注意力权重（可选）
                alphas.append(alpha)

            # 选取 top-k 的候选序列
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:k]
            sequences = [x[0] for x in all_candidates]
            scores = torch.tensor([x[1] for x in all_candidates]).to(device)
            hidden_states = [x[2] for x in all_candidates]

            # 如果所有序列都结束，提前停止
            if all(seq[-1] == word_map["<end>"] for seq in sequences):
                break

        # 返回得分最高的序列
        best_seq_idx = scores.argmax().item()
        return sequences[best_seq_idx], scores[best_seq_idx].item()  



class Model(nn.Module):

    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, encoded_captions=None, caption_lengths=None, p=1.0):
        encoder_out = self.encoder(images)  # 编码器输出
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = self.decoder(
            encoder_out, encoded_captions, caption_lengths, p
        )
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
