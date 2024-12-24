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

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# 初始化
p_max = 0.3   # 初始概率（最大值）
p_min = 0.05   # 最小概率
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
    if(epoch > epochs - 18):
        p = 0
    return p


model.device = device



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
        dropout=0.1,
        p=1.0
    )
    
        
    # 实例化模型
    mymodel = model.Model(encoder, decoder).to(device)

    # 加载模型和最佳分数
    best_score = load_model(model=mymodel,checkpoint_dir=checkpoint_dir,is_best=False)
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=word_map["<pad>"]).to(device)
    save_model_structure(mymodel)
    # 冻结解码器权重
    for param in mymodel.decoder.parameters():
        param.requires_grad = False
    #for param in mymodel.decoder.parameters():
        #param.requires_grad = False
    # 自定义的数据集
    train_dataset = MyDataset(dataset_dir, is_train=True)
    eval_dataset = MyDataset(dataset_dir, is_train=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_MyDataset,
        num_workers=4
    )
    val_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn_MyDataset,
        num_workers=4
    )
    
    # 定义编码器和解码器的参数组
    encoder_params = list(mymodel.encoder.parameters())
    decoder_params = list(mymodel.decoder.parameters())

   

    # 分别为编码器和解码器设置不同的学习率和权重衰减（正则化）
    optimizer = torch.optim.Adam(
        [
            {'params': encoder_params, 'lr': 0.0001, 'weight_decay': 1e-4},  # 编码器的学习率和正则化
            {'params': decoder_params, 'lr': 0.0002, 'weight_decay': 1e-4},  # 解码器的学习率和正则化
        ],
        betas=(0.9, 0.999), 
        eps=1e-8
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs-5, eta_min=1e-6)

    PrintFreezeLayer(mymodel)

    trainTest(train_loader=train_loader,
              val_loader=val_loader,
              model=mymodel,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=scheduler,
              num_epochs=epochs,
              p=1,
              index_word_map=index_word_map,
              best_score=best_score
            )

def trainTest(train_loader, val_loader, model, criterion, optimizer, scheduler,num_epochs, p, index_word_map, best_score):
    """
    Performs one epoch's training.
    :param train_loader: 训练集的dataloader
    :param criterion: 损失函数
    :param epoch: epoch number
    """
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0  # 记录总处理样本数
        epoch_start_time = time.time()  # 记录每个epoch的开始时间

        #if(epoch > 35):
            # 解码器权重
            #for param in model.encoder.parameters():
                #param.requires_grad = True

        p = CosineAnnealingte_p(epoch)
        
        # Batches
        with tqdm(enumerate(train_loader), total=len(train_loader)) as it:
            for i, (imgs, caps, caplens) in it:

                # 记录每个批次的开始时间
                batch_start_time = time.time()

                # Move to GPU, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)

                
                # 清空梯度
                optimizer.zero_grad()
                #decode_lengths为caplens - 1
                predictions, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs, caps, caplens , p=p)
                
                #形状为(batchsize,decode_length) 去掉了起始的<start>
                targets = caps_sorted[:, 1:]

                loss = criterion(predictions.reshape(-1, len(word_map)), targets.reshape(-1))  # (batch_size * seq_len, vocab_size)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 使用梯度裁剪限制范数

                # 更新模型参数
                optimizer.step()

                

                # 累计损失
                running_loss += loss.item()
                total_samples += imgs.size(0)

                # 累计损失
                running_loss += loss.item()
                total_samples += imgs.size(0)

                # 计算每秒样本数(SPS)
                batch_time = time.time() - batch_start_time  # 计算每个batch的时间
                sps = imgs.size(0) / batch_time  # 每秒样本数

                # 更新进度条显示
                it.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total_samples:.4f}, SPS: {sps:.2f}')

        # 更新学习率 (余弦退火)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} finished in {epoch_time:.2f} seconds")
        print(f"Average Loss for Epoch {epoch+1}: {running_loss / total_samples:.4f}")
        print(f"Teacher P : {p}")
        
        # 打印学习率
        if scheduler is not None:
            encoder_lr = scheduler.get_last_lr()[0]  # 编码器的学习率
            decoder_lr = scheduler.get_last_lr()[1]  # 解码器的学习率
            print(f"Learning Rate for Encoder: {encoder_lr:.6f}")
            print(f"Learning Rate for Decoder: {decoder_lr:.6f}")
        
        

        score = validateTest(val_loader, model, criterion, index_word_map, epoch)
        print(f"Validation Score for Epoch {epoch+1}: {score:.6f}")

        # 打印分隔线
        print('--------------------------------------------------------------------------')

        # 保存当前模型
        if score > best_score:
            # 保存最佳模型（编码器、解码器、优化器的状态）
            save_model(model=model, best_score=score, checkpoint_dir=checkpoint_dir, is_best=True)
            best_score = score  # 更新最佳分数
            
        elif epoch % 2 == 0 :
            # 保存当前模型（编码器、解码器、优化器的状态）
            save_model(model=model, best_score=score, checkpoint_dir=checkpoint_dir, is_best=False)
            save_optimizer(optimizer=optimizer,checkpoint_dir=checkpoint_dir,epoch=epoch,best_score=best_score,is_best=False)
        print('--------------------------------------------------------------------------')



def validateTest(val_loader, model, criterion, index_word_map, epoch):
    model.eval()  # Set the model to evaluation mode
    losses = AverageMeter()
    top5accs = AverageMeter()
    top1accs = AverageMeter()
    references = []
    hypotheses = []

    reflen = []
    hyplen = []
    
    with torch.no_grad():  # Disable gradient computation
        with tqdm(enumerate(val_loader), leave=False, total=len(val_loader)) as it:
            for i, (imgs, caps, caplens) in it:
                # Move to GPU if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)
                
                
                # Forward pass through the model
                predictions, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs, caps, caplens, p=0)  # no teacher forcing during eval
                targets = caps_sorted[:, 1:]  # Ignore <start> token

                # Compute loss
                loss = criterion(predictions.reshape(-1, len(word_map)), targets.reshape(-1))
                losses.update(loss.item(), imgs.size(0))

                # Get top-3 accuracy
                _, preds = predictions.topk(5, dim=2, largest=True, sorted=True)
                correct = preds == targets.unsqueeze(2)
                top5acc = correct.any(dim=2).sum(dim=1).float() / correct.size(1)
                top5accs.update(top5acc.mean().item(), imgs.size(0))

                # 计算top-1的预测
                _, preds_top1 = predictions.topk(1, dim=2, largest=True, sorted=True)
                # 检查预测的第一个元素是否与目标匹配
                correct_top1 = preds_top1.squeeze(2) == targets
                # 计算top-1的准确率
                top1acc = correct_top1.sum(dim=1).float() / correct_top1.size(1)
                # 更新top-1准确率的统计
                top1accs.update(top1acc.mean().item(), imgs.size(0))

                # Collect references and hypotheses for BLEU score
                # 收集参考答案和模型输出的预测，准备计算 BLEU 分数
                for j in range(imgs.size(0)):  # 遍历当前批次中的每一张图片
                    preds_cpu = preds[j].cpu().numpy()  # 将预测结果转换为 numpy 数组
                    caps_cpu = caps_sorted[j, 1:].cpu().numpy()  # 忽略 <start> token
                    # 选择 top-1 预测结果
                    preds_cpu = preds_cpu[:, 0]  # 选择每个时间步的第一个预测（即最可能的词）
                    # 在循环之前打印 preds_cpu 和 caps_cpu 的形状
                    #print("Shape of preds_cpu:", preds_cpu.shape)
                    #print("Shape of caps_cpu:", caps_cpu.shape)
                    # 参考答案：将目标 token 转换为单词（排除 <pad> token）
                    ref = [index_word_map[token] for token in caps_cpu if token.item() != word_map["<pad>"]]
                    #ref = [index_word_map[token] for token in caps_cpu ]
                    # 预测答案：将模型的预测 token 转换为单词（排除 <pad> token）
                    hyp = []
                    for token in preds_cpu:
                        token_id = token.item()
                        # 如果是<pad>标签，跳过
                        if token_id == word_map["<pad>"]:
                            continue
                        # 如果是<end>标签，停止解码
                        hyp.append(index_word_map[token_id])
                        if token_id == word_map["<end>"]:
                            break        
                    #print("len of ref:", len(ref))
                    reflen.append(len(ref))
                    hyplen.append(len(hyp))
                    #print("len of hyp:",len(hyp))

                    references.append(ref)  # 添加参考答案
                    hypotheses.append(hyp)  # 添加模型预测答案

        # 打开文件（文件名可以根据需要修改）
        with open("lengths_output.txt", "w") as f:
            # 写入标题行
            f.write("Epoch\tReference Length\tHypothesis Length\n")
    
            # 确保references和hypotheses列表中至少有5个元素
            if len(references) < 5 or len(hypotheses) < 5:
                print("Not enough data to randomly select five samples.")
            else:
            # 随机选择5个不同的索引
                random_indices = random.sample(range(len(references)), 5)
        
             # 写入每一行，格式化为索引、参考长度和预测长度
                for idx in random_indices:
                    f.write(f"\n{epoch}\t{len(references[idx])}\t{len(hypotheses[idx])}\n")   
                    f.write(f"reference: {references[idx]}\nhypothesis: {hypotheses[idx]}\n")
        # Calculate BLEU score
        Score = metrics.evaluate(losses, top1accs, top5accs, references, hypotheses, epoch)
        
    return Score



if __name__ == '__main__':
    main()