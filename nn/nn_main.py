from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import tqdm
from nn_DataProcess import prepare_data, build_word2vec, Data_set
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import os
from model import LSTMModel, LSTM_attention
from nn_Config import Config
from nn_eval import val_accuary


def train(train_dataloader, model, device, epoches, lr):
    """

    :param train_dataloader:
    :param model:
    :param device:
    :param epoches:
    :param lr:
    :return:
    """
    # 模型为训练模式
    model.train()
    # 将模型转化到gpu上
    model = model.to(device)
    print(model)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)  # 学习率调整
    best_acc = 0.88
    # 一个epoch可以认为是一次训练循环
    for epoch in range(epoches):
        train_loss = 0.0
        correct = 0
        total = 0

        # tqdm用在dataloader上其实是对每个batch和batch总数做的进度条
        train_dataloader = tqdm.tqdm(train_dataloader)
        # train_dataloader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epoches, 'lr:', scheduler.get_last_lr()[0]))
        # 遍历每个batch size数据
        for i, data_ in (enumerate(train_dataloader)):
            # 梯度清零
            optimizer.zero_grad()
            input_, target = data_[0], data_[1]
            # 将数据类型转化为整数
            input_ = input_.type(torch.LongTensor)
            target = target.type(torch.LongTensor)
            # 将数据转换到gpu上
            input_ = input_.to(device)
            target = target.to(device)
            # 前向传播
            output = model(input_)
            # 扩充维度
            target = target.squeeze(1)
            # 损失
            loss = criterion(output, target)
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            # print(predicted.shape)
            # 计数
            total += target.size(0)  # 此处的size()类似numpy的shape: np.shape(train_images)[0]
            # print(target.shape)
            # 计算预测正确的个数
            correct += (predicted == target).sum().item()
            # 评价指标F1、Recall、混淆矩阵
            F1 = f1_score(target.cpu(), predicted.cpu(), average='weighted')
            Recall = recall_score(target.cpu(), predicted.cpu(), average='micro')
            # CM=confusion_matrix(target.cpu(),predicted.cpu())
            postfix = {'train_loss: {:.5f},train_acc:{:.3f}%'
                       ',F1: {:.3f}%,Recall:{:.3f}%'.format(train_loss / (i + 1),
                                                            100 * correct / total, 100 * F1, 100 * Recall)}
            # tqdm pbar.set_postfix：设置训练时的输出
            train_dataloader.set_postfix(log=postfix)

        # 计算验证集的准确率
        acc = val_accuary(model, val_dataloader, device, criterion)
        # 当准确率提升时，保存模型。
        if acc > best_acc:
            best_acc = acc
            if os.path.exists(Config.model_state_dict_path) == False:
                os.mkdir(Config.model_state_dict_path)
            save_path = '{}_epoch_{}.pkl'.format("nn_model", epoch)
            print(os.path.join(Config.model_state_dict_path, save_path))
            torch.save(model, os.path.join(Config.model_state_dict_path, save_path))
        # 恢复到训练模式
        model.train()


if __name__ == '__main__':
    splist = []
    # 构建word2id词典
    word2id = {}
    with open(Config.word2id_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()  # 去掉\n \t 等
            splist.append(sp)
        word2id = dict(splist)  # 转成字典

    # 转换索引的数据类型为整数
    for key in word2id:
        word2id[key] = int(word2id[key])

    # 构建id2word
    id2word = {}
    for key, val in word2id.items():
        id2word[val] = key

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 得到数字索引表示的句子和标签
    train_array, train_lable, val_array, val_lable, test_array, test_lable = prepare_data(word2id,
                                                                                          train_path=Config.train_path,
                                                                                          val_path=Config.val_path,
                                                                                          test_path=Config.test_path,
                                                                                          seq_lenth=Config.max_sen_len)
    # 构建训练Data_set与DataLoader
    train_loader = Data_set(train_array, train_lable)
    train_dataloader = DataLoader(train_loader,
                                  batch_size=Config.batch_size,
                                  shuffle=True,
                                  num_workers=0)  # 用了workers反而变慢了
    # 构建验证Data_set与DataLoader
    val_loader = Data_set(val_array, val_lable)
    val_dataloader = DataLoader(val_loader,
                                batch_size=Config.batch_size,
                                shuffle=True,
                                num_workers=0)

    # 构建测试Data_set与DataLoader
    test_loader = Data_set(test_array, test_lable)
    test_dataloader = DataLoader(test_loader,
                                 batch_size=Config.batch_size,
                                 shuffle=True,
                                 num_workers=0)
    # 构建word2vec词向量
    w2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
    # 将词向量转化为Tensor
    w2vec = torch.from_numpy(w2vec)
    # CUDA接受float32，不接受float64
    w2vec = w2vec.float()
    # LSTM_attention
    model = LSTM_attention(Config.vocab_size, Config.embedding_dim, w2vec, Config.update_w2v,
                           Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class,
                           Config.bidirectional)

    # 训练
    train(train_dataloader, model=model, device=device, epoches=Config.n_epoch, lr=Config.lr)