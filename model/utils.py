import os
import numpy as np
import json
import cv2
import torch


def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, labels, batch_images

def collate_fn_MyDataset(
    batch: list[tuple]
):
    max_width, max_height, max_length = 0, 0, 0
    BATCH_SIZE, CHANNEL = len(batch), 1
    proper_items = []
    for sample in batch: # iterate over each samples
        image, label, label_len = sample
        if image.shape[0] * max_width > 1600 * 320 or image.shape[1] * max_height > 1600 * 320:
            continue
        max_height = max(max_height, image.shape[0])
        max_width = max(max_width, image.shape[1])
        max_length = max(max_length, label_len)
        proper_items.append(sample)

    images, image_masks = torch.zeros((len(proper_items), CHANNEL, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))
    label_lens = torch.empty((len(proper_items)))

    for i, sample in enumerate(proper_items):
        image, label, label_len = sample
        h, w = image.shape
        images[i][:, :h, :w] = image
        image_masks[i][:, :h, :w] = 1
        l = label_len
        labels[i][:l] = label
        labels_masks[i][:l] = 1
        label_lens[i] = label_len
    return images, labels, label_lens

def load_json(path):
    with open(path,'r')as f:
        data = json.load(f)
    return data

def cal_word_freq(vocab,formuladataset):
    #统计词频用于计算perplexity
    word_count = {}
    for i in vocab.values():
        word_count[i] = 0
    count = 0
    for i in formuladataset.data.values():
        words = i['caption'].split()
        for j in words:
            word_count[vocab[j]] += 1
            count += 1
    for i in word_count.keys():
        word_count[i] = word_count[i]/count
    return word_count

def get_latex_ocrdata(path,mode = 'val'):
    assert mode in ['val','train','test']
    match = []
    with open(path + 'matching/'+mode+'.matching.txt','r')as f:
        for i in f.readlines():
            match.append(i[:-1])

    formula = []
    with open(path + 'formulas/'+mode+'.formulas.norm.txt','r')as f:
        for i in f.readlines():
            formula.append(i[:-1])

    vocab_temp = set()
    data = {}

    for i in match:
        img_path = path + 'images/images_' + mode + '/' + i.split()[0]
        try:
            img = cv2.imread(img_path)
        except:
            print('Can\'t read'+i.split()[0])
            continue
        if img is None:
            continue
        size = (img.shape[1],img.shape[0])
        del img
        temp = formula[int(i.split()[1])].replace('\\n','')
        # token = set()
        for j in temp.split():
            # token.add(j)
            vocab_temp.add(j)
        data[i.split()[0]] = {'img_path':img_path,'size':size,
        'caption':temp,'caption_len':len(temp.split())+2}#这里需要加上开始以及停止符
        # data[i.split()[0]] = {'img_path':path + 'images/images_' + mode + '/' + i.split()[0],
        # 'token':list(token),'caption':temp,'caption_len':len(temp.split())+2}#这里需要加上开始以及停止符
    vocab_temp = list(vocab_temp)
    vocab = {}
    for i in range(len(vocab_temp)):
        vocab[vocab_temp[i]] = i+1
    vocab['<unk>'] = len(vocab) + 1
    vocab['<start>'] = len(vocab) + 1
    vocab['<end>'] = len(vocab) + 1
    vocab['<pad>'] = 0
    return vocab,data


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.
    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪用于避免梯度爆炸
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
    decoder_optimizer,score, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'score': score,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer':encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，自动创建
    filename = os.path.join(save_dir, 'checkpoint_' + data_name + '.pth')
    torch.save(state, filename)
    print("ok")
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        filename = os.path.join(save_dir, 'BEST_' + data_name + '.pth')
        torch.save(state, filename)


class AverageMeter(object):
    """
    一个用于跟踪变量当前值，平均值，和以及计数的对象
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


import torch
import json
import sys

def get_model_structure(model):
    """
    获取模型的结构信息（每层的类型和维度）
    :param model: 需要保存结构信息的模型
    :return: 模型结构的字典
    """
    model_structure = []

    # 遍历模型的所有子模块
    for name, layer in model.named_children():
        # 如果是 nn.Module 类型的层
        if isinstance(layer, torch.nn.Module):
            layer_info = {
                'name': name,
                'type': str(layer.__class__.__name__),  # 获取层的类型
                'parameters': get_layer_parameters(layer),  # 获取该层的参数维度
            }
            model_structure.append(layer_info)
    return model_structure

def get_layer_parameters(layer):
    """
    获取某层的输入输出维度信息（适用于常见层）
    :param layer: nn.Module 类型的单一层
    :return: 该层的参数维度信息
    """
    params = {}
    
    # 根据层的类型获取对应的参数
    if isinstance(layer, torch.nn.Linear):
        params['input_dim'] = layer.in_features
        params['output_dim'] = layer.out_features
    elif isinstance(layer, torch.nn.Conv2d):
        params['input_channels'] = layer.in_channels
        params['output_channels'] = layer.out_channels
        params['kernel_size'] = layer.kernel_size
    elif isinstance(layer, torch.nn.LSTM):
        params['input_dim'] = layer.input_size
        params['hidden_dim'] = layer.hidden_size
    elif isinstance(layer, torch.nn.GRU):
        params['input_dim'] = layer.input_size
        params['hidden_dim'] = layer.hidden_size
    elif isinstance(layer, torch.nn.Embedding):
        params['num_embeddings'] = layer.num_embeddings
        params['embedding_dim'] = layer.embedding_dim
    # 可以继续根据不同的层类型扩展
    
    return params

def save_model_structure_to_json(model, json_filename):

    """
    保存模型结构信息到 JSON 文件
    :param model: 需要保存结构信息的模型
    :param json_filename: 保存的 JSON 文件名
    """
    # 获取模型结构
    model_structure = get_model_structure(model)
    
    # 将结构信息写入 JSON 文件
    with open(json_filename, 'w') as f:
        json.dump(model_structure, f, indent=4)
    
    print(f"Model structure saved to {json_filename}")

def PrintFreezeLayer(model):
    # 打开一个文件以写入模型信息
    with open('freeze_gradient_info.txt', 'w') as f:
        sys.stdout = f
        for name, param in model.named_parameters():
            print(f"Parameter: {name} - Requires Grad: {param.requires_grad}")
        sys.stdout = sys.__stdout__  # 恢复标准输出