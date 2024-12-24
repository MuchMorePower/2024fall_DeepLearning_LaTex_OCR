#数据路径
dataset_dir = "/home/ydai/dl/dl_Project/densenet_ocr/datasets/MyDataset/"
data_name = 'MyDataset'  # 模型名称,仅在保存的时候用到
vocab_path = './data_preprocess/vocab.txt'
train_set_path = './data/small/train.json'
val_set_path = './data/small/val.json'
testDataSet_Dir = "/home/ydai/dl/dl_Project/densenet_ocr/datasets/MyDataset_test/"

# 模型参数
emb_dim = 128  # 词嵌入维数80
attention_dim = 256  # attention 层维度 256
decoder_dim = 512  # decoder维度 128
dropout = 0.5
buckets = [[240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100],
        [560, 80], [560, 100], [640, 80], [640, 100], [720, 80], [720, 100],
        [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
        [1000, 400], [1200, 200], [1600, 200],
        ]


# 训练参数
start_epoch = 0
epochs = 30  # 不触发早停机制时候最大迭代次数
epochs_since_improvement = 0  # 用于跟踪在验证集上分数没有提高的迭代次数
batch_size = 18 #训练解批大小
test_batch_size = 2 #验证集批大小
encoder_lr = 4e-4  # 学习率
decoder_lr = 8e-4  # 学习率
grad_clip = 3.  # 梯度裁剪阈值
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_score = 0.  # 目前最好的 score 
print_freq = 100  # 状态的批次打印间隔
# checkpoint = 'BEST_checkpoint_CROHME.pth.tar'  # checkpoint文件目录(用于断点继续训练)
checkpoint = None  # checkpoint文件目录(用于断点继续训练)
save_freq = 2 #保存的间隔