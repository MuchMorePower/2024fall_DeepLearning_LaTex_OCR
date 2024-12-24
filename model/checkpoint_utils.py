import os
import sys
import torch

# 在当前工作目录中创建 checkpoint 文件夹
checkpoint_dir = 'checkpoint'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

import torch
import os

# 创建保存模型的目录
checkpoint_dir = 'checkpoint'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def save_model(model, best_score, checkpoint_dir, save_encoder=True, save_decoder=True, is_best=False):
    """ 保存模型的编码器和解码器权重，同时保存最佳分数 """
    
    # 保存编码器
    if save_encoder:
        encoder_checkpoint_path = os.path.join(checkpoint_dir, 'encoder.pth')
        torch.save(model.encoder.state_dict(), encoder_checkpoint_path)
        print(f"Encoder checkpoint saved at {encoder_checkpoint_path}")
    
    # 保存解码器
    if save_decoder:
        decoder_checkpoint_path = os.path.join(checkpoint_dir, 'decoder.pth')
        torch.save(model.decoder.state_dict(), decoder_checkpoint_path)
        print(f"Decoder checkpoint saved at {decoder_checkpoint_path}")
    
    
    
    # 如果是最佳模型，保存为 best_model.pth（包含编码器和解码器）
    if is_best:
        # 保存最佳分数
        best_score_path = os.path.join(checkpoint_dir, 'best_score.pth')
        torch.save({'best_score': best_score}, best_score_path)
        print(f"Best score saved at {best_score_path}")
        
        best_encoder_path = os.path.join(checkpoint_dir, 'best_encoder.pth')
        best_decoder_path = os.path.join(checkpoint_dir, 'best_decoder.pth')
        
        torch.save(model.encoder.state_dict(), best_encoder_path)
        torch.save(model.decoder.state_dict(), best_decoder_path)
        
        print(f"Best encoder model saved at {best_encoder_path}")
        print(f"Best decoder model saved at {best_decoder_path}")


def load_model(model, checkpoint_dir, is_best=False, load_encoder=True, load_decoder=True):
    """ 加载编码器和解码器的权重，同时加载最佳分数 """
    
    # 根据是否加载最佳模型，选择不同的路径
    if is_best:
        encoder_checkpoint_path = os.path.join(checkpoint_dir, 'best_encoder.pth')
        decoder_checkpoint_path = os.path.join(checkpoint_dir, 'best_decoder.pth')
    else:
        encoder_checkpoint_path = os.path.join(checkpoint_dir, 'encoder.pth')
        decoder_checkpoint_path = os.path.join(checkpoint_dir, 'decoder.pth')

    # 加载编码器
    if load_encoder and os.path.exists(encoder_checkpoint_path):
        model.encoder.load_state_dict(torch.load(encoder_checkpoint_path),strict=False)
        print(f"Encoder loaded from {encoder_checkpoint_path}")
    else:
        print(f"Encoder not loaded from {encoder_checkpoint_path}")
    
    # 加载解码器
    if load_decoder and os.path.exists(decoder_checkpoint_path):
        model.decoder.load_state_dict(torch.load(decoder_checkpoint_path), strict=False)
        print(f"Decoder loaded from {decoder_checkpoint_path}")
    else:
        print(f"Decoder not loaded from {decoder_checkpoint_path}")
    
    # 加载最佳分数
    best_score_path = os.path.join(checkpoint_dir, 'best_score.pth')
    if os.path.exists(best_score_path):
        checkpoint = torch.load(best_score_path)
        best_score = checkpoint['best_score']
        print(f"Best score loaded from {best_score_path}: {best_score}")
    else:
        best_score = float('-inf')  # 如果没有找到最佳分数，则设置为负无穷
        print("Best score not found. Initializing as -inf.")

    
    return best_score


def load_optimizer(optimizer, checkpoint_path):
    """ 加载优化器的状态 """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
        epoch = checkpoint['epoch']
        best_score = checkpoint['best_score']
        print(f"Optimizer state loaded from {checkpoint_path}")
        return epoch, best_score
    else:
        print(f"No checkpoint found for optimizer state at {checkpoint_path}.")
        return 0, float('inf')  # 如果没有找到优化器的状态，返回初始值

def save_optimizer(optimizer, checkpoint_dir, epoch, best_score, is_best=False):
    """ 保存优化器的状态，以及训练的 epoch 和最佳分数 """
    
    # 保存优化器的状态字典
    optimizer_checkpoint_path = os.path.join(checkpoint_dir, 'optimizer.pth')
    checkpoint = {
        'epoch': epoch,
        'best_score': best_score,
        'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器的状态
    }
    torch.save(checkpoint, optimizer_checkpoint_path)
    print(f"Optimizer state saved to {optimizer_checkpoint_path}")

    # 如果是最佳模型，保存最佳优化器状态
    if is_best:
        best_optimizer_checkpoint_path = os.path.join(checkpoint_dir, 'best_optimizer.pth')
        torch.save(checkpoint, best_optimizer_checkpoint_path)
        print(f"Best optimizer state saved to {best_optimizer_checkpoint_path}")

def save_model_structure(model):
    # 打开一个文件以写入模型信息
    with open('model_info.txt', 'w') as f:
        sys.stdout = f
        print(model)  # 打印模型信息到文件
        sys.stdout = sys.__stdout__  # 恢复标准输出
    
def save_transfer_model(model, best_score, checkpoint_dir, save_encoder=True, save_decoder=True, is_best=False):
    """ 保存模型的编码器和解码器权重，同时保存最佳分数 """
    
    # 保存编码器
    if save_encoder:
        encoder_checkpoint_path = os.path.join(checkpoint_dir, 'encoder.pth')
        torch.save(model.encoder.state_dict(), encoder_checkpoint_path)
        print(f"Encoder checkpoint saved at {encoder_checkpoint_path}")
    
    # 保存解码器
    if save_decoder:
        decoder_checkpoint_path = os.path.join(checkpoint_dir, 'decoder.pth')
        torch.save(model.decoder.state_dict(), decoder_checkpoint_path)
        print(f"Decoder checkpoint saved at {decoder_checkpoint_path}")
    
    
    
    # 如果是最佳模型，保存为 best_model.pth（包含编码器和解码器）
    if is_best:
        # 保存最佳分数
        best_score_path = os.path.join(checkpoint_dir, 'best_score.pth')
        torch.save({'best_score': best_score}, best_score_path)
        print(f"Best score saved at {best_score_path}")
        if save_encoder:
            best_encoder_path = os.path.join(checkpoint_dir, 'best_encoder.pth')
            torch.save(model.encoder.state_dict(), best_encoder_path)
            print(f"Best encoder model saved at {best_encoder_path}")
        
        if save_decoder:
            best_decoder_path = os.path.join(checkpoint_dir, 'best_decoder.pth')
            torch.save(model.decoder.state_dict(), best_decoder_path)
            print(f"Best decoder model saved at {best_decoder_path}")

def load_tranfer_model(model,checkpoint_dir,train_encoder=False,train_decoder=False):

    if train_encoder:
        encoder_checkpoint_path = os.path.join(checkpoint_dir, 'encoder.pth')
        decoder_checkpoint_path = os.path.join(checkpoint_dir, 'best_decoder.pth')
    elif train_decoder:
        encoder_checkpoint_path = os.path.join(checkpoint_dir, 'best_encoder.pth')
        decoder_checkpoint_path = os.path.join(checkpoint_dir, 'decoder.pth')

    if os.path.exists(encoder_checkpoint_path):
        model.encoder.load_state_dict(torch.load(encoder_checkpoint_path),strict=False)
        print(f"Encoder loaded from {encoder_checkpoint_path}")
    else:
        print(f"Encoder not loaded from {encoder_checkpoint_path}")


    if os.path.exists(decoder_checkpoint_path):
        model.decoder.load_state_dict(torch.load(decoder_checkpoint_path), strict=False)
        print(f"Decoder loaded from {decoder_checkpoint_path}")
    else:
        print(f"Decoder not loaded from {decoder_checkpoint_path}")

    # 加载最佳分数
    best_score_path = os.path.join(checkpoint_dir, 'best_score.pth')
    if os.path.exists(best_score_path):
        checkpoint = torch.load(best_score_path)
        best_score = checkpoint['best_score']
        print(f"Best score loaded from {best_score_path}: {best_score}")
    else:
        best_score = float('-inf')  # 如果没有找到最佳分数，则设置为负无穷
        print("Best score not found. Initializing as -inf.")

    
    return best_score


        
