import torch
import jieba

def read_word2id(file_path):
    word_to_idx = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                word, idx = line.split()
                word_to_idx[word] = int(idx)
    return word_to_idx

# 使用示例：
word_to_idx = read_word2id('../word2vec_data/word2id.txt')
# print(word_to_idx)

def classify_review(comment, model):
    words = list(jieba.cut(comment, cut_all=True))
    word_indices = [word_to_idx[word] if word in word_to_idx else word_to_idx['_PAD_'] for word in words]

    # 将词索引转换为张量
    words_tensor = torch.tensor(word_indices).unsqueeze(0)
    # print("words_tensor: ", words_tensor)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)
        words_tensor = words_tensor.to(device)

    # 使用模型进行分类
    with torch.no_grad():
        embeddings = model.embedding(words_tensor)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        states, hidden = model.encoder(embeddings)  # [batch, seq_len, embed_dim]
        output = model.decoder1(states[:, -1, :])
        output = model.decoder2(output)
        # print("output:", output)
        _, pred = torch.max(output, 1)
        print("nn_prediction: ", pred)

    if pred.item() == 0:
        # print("好评")
        return 0
    elif pred.item() == 1:
        # print("坏评")
        return 1