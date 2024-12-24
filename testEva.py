import time
from config import *
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from model.utils import *
from model import metrics,dataloader
from model import model
from model.checkpoint_utils import *
from torch.utils.checkpoint import checkpoint as train_ck
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.dataloader import MyDataset

import json
import math
import random
import numpy as np
import csv  # 导入csv模块
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化
p_max = 0.9   # 初始概率（最大值）
p_min = 0.1   # 最小概率
T_max = epochs - 20   # 总训练轮数

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作的结果一致
    torch.backends.cudnn.benchmark = False  # 关闭cuDNN的自动优化功能

# 计算当前 epoch 对应的概率
def CosineAnnealingte_p(epoch):
    p = p_min + 0.5 * (p_max - p_min) * (1 + math.cos(math.pi * epoch / T_max))
    if(epoch > epochs - 10):
        p = 0
    return p





model.device = device
'''
如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
'''
# cudnn.benchmark = True


def main():
    """
    Training and validation.
    """
    
    # 设置随机种子
    set_seed(82)
    
    
    global best_score, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # 字典文件
    # word_map = load_json(vocab_path)
    with open(vocab_path) as f:
        words = f.readlines()
    words.append("<start>")
    words.append("<end>")
    word_map = {value: index + 1 for index, value in enumerate(words)}
    word_map["<pad>"] = 0
    # 保存 word_map 到 JSON 文件
    with open('word_map.json', 'w') as f:
        json.dump(word_map, f, indent=4)  # indent=4 用于美化格式
    # 从 word_map.json 文件中读取 word_map
    with open('word_map.json', 'r') as f:
        word_map = json.load(f)

    # 反转字典
    index_word_map = {v: k for k, v in word_map.items()}
    

    encoder = model.Encoder()

    decoder = model.DecoderWithAttention(
        attention_dim=256,
        embed_dim=128,
        decoder_dim=512,
        vocab_size=len(word_map),
        encoder_dim=512,
        dropout=0.2,
        p=1.0
    )
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
        
    # 实例化模型
    mymodel = Model(encoder, decoder).to(device)

    # 加载模型和最佳分数
    best_score = load_model(mymodel, checkpoint_dir, is_best=True)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"]).to(device)

    # 自定义的数据集
    eval_dataset = MyDataset(testDataSet_Dir
                             , is_train=False)

    test_loader = DataLoader(
        eval_dataset, 
        batch_size=test_batch_size,
        collate_fn=collate_fn_MyDataset,
        num_workers=2
    )

    evaluateTest(test_loader,mymodel,criterion,index_word_map)    




def evaluateTest(val_loader, model, criterion, index_word_map, epoch=0):
    model.eval()  # Set the model to evaluation mode
    losses = AverageMeter()
    top5accs = AverageMeter()
    top1accs = AverageMeter()
    references = []
    hypotheses = []

    with torch.no_grad():  # Disable gradient computation
        with tqdm(enumerate(val_loader), leave=False, total=len(val_loader)) as it:
            for i, (imgs, *_) in it:  # 忽略 caps 和 caplens
                # Move to GPU if available
                imgs = imgs.to(device)
                
                # Forward pass through the model
                predictions, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs, None, None, p=0)  # no teacher forcing during eval
                
                # Get top-3 accuracy
                _, preds = predictions.topk(5, dim=2, largest=True, sorted=True)
                
                # 收集模型输出的预测，准备保存到文件
                for j in range(imgs.size(0)):  # 遍历当前批次中的每一张图片
                    preds_cpu = preds[j].cpu().numpy()  # 将预测结果转换为 numpy 数组
                    # 选择 top-1 预测结果
                    preds_cpu = preds_cpu[:, 0]  # 选择每个时间步的第一个预测（即最可能的词）
                    # 预测答案：将模型的预测 token 转换为单词（排除 <pad> token）
                    hyp = []
                    for token in preds_cpu:
                        token_id = token.item()
                        # 如果是<pad>标签，跳过
                        if token_id == word_map["<pad>"]:
                            continue
                        # 如果是<end>标签，停止解码
                        
                        if token_id == word_map["<end>"]:
                            break  
                        hyp.append(index_word_map[token_id])               
                    hypotheses.append(hyp)  # 添加模型预测答案

        with open("model1_pred_v6.txt", "w", encoding="utf-8") as f:
            for idx in range(len(hypotheses)):
                # 去除每个元素中的换行符并拼接
                formatted_sentence = ' '.join([word.strip() for word in hypotheses[idx]])  # 移除换行符并拼接
                f.write(f"{formatted_sentence}\n")  # 每个句子之后换行


        



if __name__ == '__main__':
    main()