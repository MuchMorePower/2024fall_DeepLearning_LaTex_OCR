import os
from pathlib import Path

import torchvision
import torch
import json
import cv2
import numpy as np
from scipy.ndimage import zoom

from config import vocab_path,buckets
from torch.utils.data import Dataset
from model.utils import load_json

with open(vocab_path) as f:
    words = f.readlines()
words.append("<start>")
words.append("<end>")
vocab = {value.strip(): index + 1 for index, value in enumerate(words)}
vocab["<pad>"] = 0

def get_new_size(old_size, buckets=buckets,ratio = 2):
    """Computes new size from buckets

    Args:
        old_size: (width, height)
        buckets: list of sizes

    Returns:
        new_size: original size or first bucket in iter order that matches the
            size.

    """
    if buckets is None:
        return old_size
    else:
        w, h = old_size[0]/ratio,old_size[1]/ratio
        for (idx,(w_b, h_b)) in enumerate(buckets):
            if w_b >= w and h_b >= h:
                return w_b, h_b,idx

    return old_size

def data_turn(img_data,pad_size = [8,8,8,8],resize = False):
    #找到字符区域边界
    nnz_inds = np.where(img_data != 255)
    y_min = np.min(nnz_inds[1])
    y_max = np.max(nnz_inds[1])
    x_min = np.min(nnz_inds[0])
    x_max = np.max(nnz_inds[0])
    old_im = img_data[x_min:x_max+1,y_min:y_max+1]

    #pad the image
    top, left, bottom, right = pad_size
    old_size = (old_im.shape[0] + left + right, old_im.shape[1] + top + bottom)
    new_im = np.ones(old_size , dtype = np.uint8)*255
    new_im[top:top+old_im.shape[0],left:left+old_im.shape[1]] = old_im
    if resize:
        new_size = get_new_size(old_size, buckets)[:2]
        new_im = cv2.resize(new_im,new_size, cv2.INTER_LANCZOS4)
    return new_im


def label_transform(text,start_type = '<start>',end_type = '<end>',pad_type = '<pad>',max_len = 160):
    text = text.split()
    text = [start_type] + text + [end_type]
    # while len(text)<max_len:
    #     text += [pad_type]
    # print(f"{vocab=}")
    # 转换为索引，未知词用 pad_type 替代
    text = [vocab.get(word, vocab[pad_type]) for word in text]
    #text = [i for i in map(lambda x:vocab[x],text)]
    return text
    # return torch.LongTensor(text)

def img_transform(img,size,ratio = 1):
    #downsample
    new_size = (int(img.shape[1]/ratio), int(img.shape[0]/ratio))
    new_im = cv2.resize(img,new_size, cv2.INTER_LANCZOS4)#先进行下采样
    new_im = cv2.resize(img,tuple(size))#再缩放到需要的大小
    new_im = new_im[:,:,np.newaxis]
    to_tensor = torchvision.transforms.ToTensor()
    return to_tensor(new_im)

class MyDataset(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        img_transform=img_transform,
        label_transform = label_transform,
        ratio = 2,
        is_train = True
    ):
        self.dataset_dir = dataset_dir
        self.img_transform = img_transform # 传入图片预处理
        self.label_transform = label_transform # 传入图片预处理
        self.ratio = ratio#下采样率

        # list all npy files under the directory
        npy_files_list = []
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith(".npy"):
                npy_files_list.append(file_name)
        split_dict = {
            "train": (0, 0.9),
            "val": (0.9, 1),
        }
        split_tuple = split_dict["train"] if is_train else split_dict["val"]
        n_files = len(npy_files_list)
        self.npy_files_list = npy_files_list[int(split_tuple[0] * n_files): int(split_tuple[1] * n_files)]
        self.n_files = len(self.npy_files_list)
        print(f"files {self.npy_files_list} are used as {'train' if is_train else 'val'} set")

        # determine the number of samples as delimiters; n files has n+1 delimiters
        self.n_samples_list = [0]
        if len(self.npy_files_list) == 0:
            print(f"No .npy file found under the directory {Path(dataset_dir)}")
            exit(1)
        for file_name in self.npy_files_list:
            temp_images_and_labels: list[dict] = np.load(Path(dataset_dir) / file_name, allow_pickle=True)
            self.n_samples_list.append(self.n_samples_list[-1] + len(temp_images_and_labels))
        del temp_images_and_labels

        # record the current loaded file index in self.npy_file_list
        self.current_file_idx = 0
        # init with the first file
        self.images_and_labels: list[dict] = np.load(Path(dataset_dir) / self.npy_files_list[0], allow_pickle=True)

    def __getitem__(self, idx):
        # WARNING: the codes assume shuffle=False

        # whether to load next .npy file
        if idx >= self.n_samples_list[self.current_file_idx + 1]:
            self.images_and_labels = np.load(Path(self.dataset_dir) / self.npy_files_list[self.current_file_idx + 1], allow_pickle=True)
            self.current_file_idx += 1
        
        idx_in_list = idx - self.n_samples_list[self.current_file_idx]
        image = self.images_and_labels[idx_in_list]["image"]
        if image.shape[-1] == 3:
            # convert RGB to grayscale and normalize
            # (w, h, c) -> (w, h)
            image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]) / 255
        # Perform downsampling using scipy's zoom function
        image = zoom(image, 1 / self.ratio, order=1)  # order=1 corresponds to bilinear interpolation
        image = torch.tensor(image).float()
        label: str = self.images_and_labels[idx_in_list]["label"]
        label_list = self.label_transform(label)

        # reset the counter and init between epochs
        if idx == self.n_samples_list[-1] - 1:
            self.current_file_idx = 0
            self.images_and_labels = np.load(Path(self.dataset_dir) / self.npy_files_list[0], allow_pickle=True)

        return image, torch.LongTensor(label_list), torch.tensor([len(label_list)])

    def __len__(self):
        #返回每个epoch训练的样本总数量
        return self.n_samples_list[-1] #正常训练用
        #return min(self.n_samples_list[-1],10)  # 每个 epoch 固定只用 10 个样本,debug用

class formuladataset(object):
    '''
    公式数据集,负责读取图片和标签,同时自动对进行预处理
    ：param json_path 包含图片文件名和标签的json文件
    ：param pic_transform,label_transform分别是图片预处理和标签预处理(主要是padding)
    '''
    def __init__(self, data_json_path, img_transform=img_transform,label_transform = label_transform,ratio = 2,batch_size = 2):
        self.img_transform = img_transform # 传入图片预处理
        self.label_transform = label_transform # 传入图片预处理
        self.data = load_json(data_json_path)#主要的数据文件
        self.ratio = ratio#下采样率
        self.batch_size = batch_size#批大小
        self.buckets = buckets#尺寸分类
        self.buckets_index = np.array([i for i in range(len(self.buckets))],dtype = np.int32)#尺寸索引,用于shuffle
        self.bucket_data = [[]for i in range(len(self.buckets))]#用于存放不同尺寸的data
        self.img_list = np.array([i for i in self.data.keys()]) # 得到所有的图像名字的列表
        # if self.batch_size!=1:
        self.bucket()
        self.iter = self._iter()

    def bucket(self):
        print('Bucking data...')
        for i,j in self.data.items():
            new_size = get_new_size(j['size'],self.buckets,self.ratio)
            if (len(new_size)!=3):
                continue
            idx = new_size[-1]
            self.bucket_data[idx].append(i)
        self.bucket_data = np.array(self.bucket_data)
        print('finish bucking!')
    
    def shuffle(self):#打乱顺序
        # if(self.batch_size==1):
        #     np.random.shuffle(self.bucket_data)
        #     self.iter = self._iter()
        # else:
        np.random.shuffle(self.buckets_index)
        self.buckets = np.array(self.buckets)
        self.buckets = self.buckets[self.buckets_index]
        self.bucket_data = self.bucket_data[self.buckets_index]
        for i in self.bucket_data:
            np.random.shuffle(i)#打乱数据的顺序
        self.iter = self._iter()

    def _iter(self):
        for size_idx,i in enumerate(self.bucket_data):
            img_batch,cap_batch,cap_len_batch = [],[],torch.zeros((self.batch_size)).int()
            idx = 0
            for j in i:
                item = self.data[j]
                caption = item['caption']
                img = cv2.imread(item['img_path'])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#图片由BGR转灰度
                cap_len_batch[idx] = item['caption_len']
                if self.img_transform is not None:
                    img = self.img_transform(img,size = self.buckets[size_idx],ratio = self.ratio)
                if self.label_transform is not None:
                    caption = self.label_transform(caption)
                img_batch.append(img.unsqueeze(dim = 0))
                cap_batch.append(caption)
                idx += 1
                if idx%self.batch_size == 0:
                    if len(img_batch)==0:
                        break
                    for ii in range(len(cap_batch)):
                        cap_batch[ii] += [vocab['<pad>']]*(int(max(cap_len_batch))-len(cap_batch[ii]))
                    cap_batch = torch.LongTensor(cap_batch)
                    yield torch.cat(img_batch,dim = 0),cap_batch,cap_len_batch
                    img_batch,cap_batch,cap_len_batch = [],[],torch.zeros(self.batch_size).int()
                    idx = 0
            if len(img_batch)==0:
                continue
            for ii in range(len(cap_batch)):
                cap_batch[ii] += [vocab['<pad>']]*(int(max(cap_len_batch))-len(cap_batch[ii]))
            cap_batch = torch.LongTensor(cap_batch)
            yield torch.cat(img_batch,dim = 0),cap_batch,cap_len_batch[:idx]
                
    def __iter__(self):
        return self.iter

    def __len__(self): # 总数据的多少
        count = 0
        for i in self.bucket_data:
            count += np.ceil(len(i)/self.batch_size)
        return int(count)