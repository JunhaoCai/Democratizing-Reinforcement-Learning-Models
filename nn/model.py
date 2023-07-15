"""
模型部分
"""
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight, update_w2v, hidden_dim,
                 num_layers, drop_keep_prob, n_class, bidirectional, **kwargs):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层节点数
        self.num_layers = num_layers  # 神经元层数
        self.n_class = n_class  # 类别数

        self.bidirectional = bidirectional  # 控制是否为双向LSTM
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)  # 读取预训练好的参数
        self.embedding.weight.requires_grad = update_w2v  # 控制加载的预训练模型在训练中参数是否更新
        # LSTM
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=drop_keep_prob)
        # 解码部分
        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):
        """
        前向传播
        :param inputs: [batch, seq_len]
        :return:
        """
        # [batch, seq_len] => [batch, seq_len, embed_dim][64,75,50]
        embeddings = self.embedding(inputs)
        # [batch, seq_len, embed_dim] = >[seq_len, batch, embed_dim]
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        # states.shape= torch.Size([65, 64, 200])
        encoding = torch.cat([states[0], states[-1]], dim=1)
        # encoding.shape= torch.Size([64, 400])
        # 解码
        outputs = self.decoder1(encoding)
        # outputs = F.softmax(outputs, dim=1)
        outputs = self.decoder2(outputs)
        return outputs


class LSTM_attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight, update_w2v, hidden_dim,
                 num_layers, drop_keep_prob, n_class, bidirectional, **kwargs):
        super(LSTM_attention, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层节点数
        self.num_layers = num_layers  # 神经元层数
        self.n_class = n_class  # 类别数

        self.bidirectional = bidirectional  # 控制是否双向LSTM
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)  # 读取预训练好的参数
        self.embedding.weight.requires_grad = update_w2v  # 控制加载的预训练模型在训练中参数是否更新
        # LSTM
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=drop_keep_prob, batch_first=True)

        # weiht_w即为公式中的h_s(参考系)
        # nn. Parameter的作用是参数是需要梯度的
        self.weight_W = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))

        # 对weight_W、weight_proj进行初始化
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):
        """
        前向传播
        :param inputs: [batch, seq_len]
        :return:
        """
        # 编码
        embeddings = self.embedding(inputs)  # [batch, seq_len] => [batch, seq_len, embed_dim][64,65,50]
        # 经过LSTM得到输出，state是一个输出序列
        # 结合batch_first设置
        states, hidden = self.encoder(embeddings.permute([0, 1, 2]))  # [batch, seq_len, embed_dim]
        # print("states.shape=", states.shape)  (64,50,200)

        # attention
        # states与self.weight_W矩阵相乘，然后做tanh
        u = torch.tanh(torch.matmul(states, self.weight_W))
        # u与self.weight_proj矩阵相乘,得到score
        att = torch.matmul(u, self.weight_proj)
        # softmax
        att_score = F.softmax(att, dim=1)
        # 加权求和
        scored_x = states * att_score
        encoding = torch.sum(scored_x, dim=1)
        # 线性层
        outputs = self.decoder1(encoding)
        outputs = self.decoder2(outputs)
        return outputs
