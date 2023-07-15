from __future__ import unicode_literals, print_function, division
from io import open
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import os
from model import LSTMModel, LSTM_attention
from nn_Config import Config
from nn_DataProcess import prepare_data, build_word2vec, text_to_array_nolable, Data_set


def val_accuary(model, val_dataloader, device, criterion):
    # # 验证模式，验证时将模型固定
    model.eval()
    # 将模型转换到gpu
    model = model.to(device)
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        val_loss = 0.0
        for j, data_1 in (enumerate(val_dataloader, 0)):
            input1, target1 = data_1[0], data_1[1]
            input1 = input1.type(torch.LongTensor)
            target1 = target1.type(torch.LongTensor)
            target1 = target1.squeeze(1)  # 从[64,1]到[64]
            input1 = input1.to(device)
            target1 = target1.to(device)
            output1 = model(input1)
            loss1 = criterion(output1, target1)
            val_loss += loss1.item()
            _, predicted1 = torch.max(output1, 1)
            total1 += target1.size(0)  # 此处的size()类似numpy的shape: np.shape(train_images)[0]
            correct1 += (predicted1 == target1).sum().item()
            F1 = f1_score(target1.cpu(), predicted1.cpu(), average='weighted')
            Recall = recall_score(target1.cpu(), predicted1.cpu(), average='micro')
            # CM = confusion_matrix(target1.cpu(), predicted1.cpu())
        print(
            '\nVal accuracy : {:.3f}%,val_loss:{:.3f}, F1_score：{:.3f}%, Recall：{:.3f}%'.format(100 * correct1 / total1,
                                                                                                val_loss, 100 * F1,
                                                                                                100 * Recall))
        return 100 * correct1 / total1


def test_accuary(model, test_dataloader, device):
    model = model.to(device)
    # 被它包括起来的部分，梯度不在更新
    with torch.no_grad():
        correct = 0
        total = 0
        # 迭代test_dataloader中的batch大小数据
        for k, data_test in (enumerate(test_dataloader, 0)):
            input_test, target_ = data_test[0], data_test[1]
            # 转换成整数
            input_test = input_test.type(torch.LongTensor)
            target_ = target_.type(torch.LongTensor)
            # 从[64,1]到[64]
            target_ = target_.squeeze(1)
            # 转换到gpu上
            input_test = input_test.to(device)
            target_ = target_.to(device)
            # 前向传播
            output2 = model(input_test)
            _, predicted_test = torch.max(output2, 1)
            # 记录总数
            total += target_.size(0)  # 此处的size()类似numpy的shape: np.shape(train_images)[0]
            # 记录正确数
            correct += (predicted_test == target_).sum().item()
            # 评价指标
            F1 = f1_score(target_.cpu(), predicted_test.cpu(), average='weighted')
            Recall = recall_score(target_.cpu(), predicted_test.cpu(), average='micro')
            CM = confusion_matrix(target_.cpu(), predicted_test.cpu())
        print('test accuracy : {:.3f}%, F1_score：{:.3f}%, Recall：{:.3f}%,Confusion_matrix：{}'.format(
            100 * correct / total, 100 * F1, 100 * Recall, CM))


def pre(word2id, model, seq_lenth, path):
    model.to("cpu")
    with torch.no_grad():
        # 加载无标签数据
        input_array = text_to_array_nolable(word2id, seq_lenth, path)
        # sen_p = sen_p.type(torch.LongTensor)
        # 转换数据类型
        sen_p = torch.from_numpy(input_array)
        sen_p = sen_p.type(torch.LongTensor)
        # 前向传播
        output_p = model(sen_p)
        _, pred = torch.max(output_p, 1)
        for i in pred:
            print('预测类别为', i.item())


if __name__ == '__main__':
    splist = []
    # 构建word2id
    word2id = {}
    with open(Config.word2id_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()  # 去掉\n \t 等
            splist.append(sp)
        word2id = dict(splist)  # 转成字典

    # 将索引转为整数
    for key in word2id:  # 将字典的值，从str转成int
        word2id[key] = int(word2id[key])

    # 转换设备到gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 得到句子id表示和标签
    train_array, train_lable, val_array, val_lable, test_array, test_lable = prepare_data(word2id,
                                                                                          train_path=Config.train_path,
                                                                                          val_path=Config.val_path,
                                                                                          test_path=Config.test_path,
                                                                                          seq_lenth=Config.max_sen_len)
    # 构建测试Data_set与DataLoader
    test_loader = Data_set(test_array, test_lable)
    test_dataloader = DataLoader(test_loader,
                                 batch_size=Config.batch_size,
                                 shuffle=True,
                                 num_workers=0)
    # 构建word2vec词向量
    w2vec = build_word2vec(Config.pre_word2vec_path,
                           word2id,
                           None)
    # 将词向量转化为Tensor
    w2vec = torch.from_numpy(w2vec)
    # CUDA接受float32，不接受float64
    w2vec = w2vec.float()
    # LSTM_attention
    model = LSTM_attention(Config.vocab_size, Config.embedding_dim, w2vec, Config.update_w2v,
                           Config.hidden_dim, Config.num_layers, Config.drop_keep_prob, Config.n_class,
                           Config.bidirectional)
    # 读取训练好的模型
    # model1 = torch.load(Config.model_state_dict_path)
    model = torch.load('../nn/nn_models/nn_model_epoch_12.pkl')

    # model.load_state_dict(torch.load(Config.model_state_dict_path)) #仅保存参数
    # 验证
    # val_accuary(model1, val_dataloader, device)
    # 测试
    test_accuary(model, test_dataloader, device)
    # 预测
    pre(word2id, model, Config.max_sen_len, Config.pre_path)
