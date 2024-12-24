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
    eval_dataset = MyDataset(dataset_dir
                             , is_train=False)

    test_loader = DataLoader(
        eval_dataset, 
        batch_size=1,
        collate_fn=collate_fn_MyDataset,
        num_workers=2
    )

    evaluateTest(test_loader,mymodel,criterion,index_word_map, word_map)    




def evaluateTest(val_loader, model, criterion, index_word_map, word_map, beam_size=5, max_caption_length=70, epoch=0):
    model.eval()  # Set the model to evaluation mode
    losses = AverageMeter()
    top5accs = AverageMeter()
    top1accs = AverageMeter()
    references = []
    hypotheses = []

    with torch.no_grad():  # 禁用梯度计算
        with tqdm(enumerate(val_loader), leave=False, total=len(val_loader)) as it:
            for i, (imgs, caps, caplens) in it:
                # 将数据移动到设备
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                # 编码图像
                encoder_out = model.encoder(imgs)  # 假设模型有编码器
                
                # 使用束搜索生成描述
                # 因为 beam_search 仅支持 batch_size=1，确保 val_loader 的 batch_size=1
                if encoder_out.size(0) != 1:
                    raise ValueError("evaluateTest 目前只支持 batch_size=1")

                best_seq,_ = model.decoder.beam_search_v0(
                    encoder_out,
                    word_map,
                    beam_size=beam_size
                )

               # 将词索引转换为单词
                hyp = []
                for token_id in best_seq:
                    token_id = token_id.item() if isinstance(token_id, torch.Tensor) else token_id  # 确保 token_id 是整数
                    if token_id == word_map["<pad>"]:
                        continue
                    hyp.append(index_word_map[token_id])  # 使用整数作为索引
                    if token_id == word_map["<end>"]:
                        break


                # 参考答案：将目标 token 转换为单词（排除 <pad> token）
                ref = [index_word_map[token] for token in caps[0].cpu().numpy() if token != word_map["<pad>"]]

                references.append(ref)  # BLEU 需要嵌套列表
                hypotheses.append(hyp)    # 添加模型预测答案


        # 打开文件（文件名可以根据需要修改）
        with open("wrong_output.txt", "w") as f:
            f.write("Epoch\tReference Length\tHypothesis Length\n")

            # 过滤出不完全匹配的句子
            mismatched_indices = [idx for idx in range(len(references)) if references[idx] != hypotheses[idx]]

            if len(mismatched_indices) < 20:
                print("Not enough mismatched sentences. Only", len(mismatched_indices), "available.")

            selected_indices = random.sample(mismatched_indices, min(20, len(mismatched_indices)))

            for idx in selected_indices:
                f.write(f"\n{epoch}\t{len(references[idx])}\t{len(hypotheses[idx])}\n")
                f.write(f"reference: {references[idx]}\nhypothesis: {hypotheses[idx]}\n")

        # Calculate BLEU score
        Score = metrics.evaluate(losses, top1accs, top5accs, references, hypotheses, epoch)

    return Score



if __name__ == '__main__':
    main()