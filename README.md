SCUT 2024 Fall Nerual NetWork and DeepLearning
把图片格式的数学表达式转化为文字格式的LaTex表达式
编码器-解码器结构，编码器CNN，解码器RNN
编码器采用的是ResNet + DenseNet，引入了CBAM模块和SE模块
解码器采用的是GRU，通过残差连接与层归一化实现了三层GRU的训练
注意力机制使用的是加性注意力，配合门控机制实现了自适应注意力
1. BLEU: 衡量模型输出文本的质量
2. EditDistance: 即 Levenshtein 距离，以取反的百分数呈现，越大越好。例：80% 的
EditDistance 代表需要改动 20% 的内容才能达到 groundtruth
3. ExactMatch: 当预测结果和 groundtruth 一样时才算 100% 准确，否则为 0%，因此同样越大越
好。
4. Model Complexity：模型进行推理需要多少的相乘相加运算
