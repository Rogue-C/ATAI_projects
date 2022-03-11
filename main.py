import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re

w2vmodel_path = "w2v(6).model"
finalmodel_path = "final(6).model"
# 定义读取数据文件的函数

DataPath = "E:\\PyCharm_File\\ATAI_project\\DataFile"
# 设置警告忽略，将一些无用的风险给忽略掉（仅仅是满足强迫症）
warnings.filterwarnings("ignore")


def read_training_data():
    # 打开训练数据，此文件里没有label
    # 每一条句子合成一行，没有id和表头
    with open(f"{DataPath}\\trainData\\reviews.txt", "r", encoding='UTF-8') as f:
        lines = f.readlines()  # 将数据按每行提取出来,lines是一个二维数组
        lines = [line.strip("\n") for line in lines]  # 以换行符作为行的分离，以空格作为单词间的分离
    x = [line[:] for line in lines]
    x = [re.sub(r"([.!?,'])", r"", s) for s in x]  # 去除掉一些标点符号
    x = [' '.join(s.split()) for s in x]
    x = [s.split() for s in x]

    # 打开train文件对应的label数据文件
    # 每一行对应train文件每一个句子的label，没有id和表头
    with open(f"{DataPath}\\trainData\\newlabels.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip("\n") for line in lines]
    x_label = [line[:] for line in lines]
    return x, x_label


def read_testing_data():
    # 打开训练数据，此文件里没有label
    # 每一条句子合成一行，没有id和表头
    with open(f"{DataPath}\\testData\\data.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip("\n") for line in lines]
    x = [line[:] for line in lines]
    x = [re.sub(r"([.!?,'])", r"", s) for s in x]
    x = [' '.join(s.split()) for s in x]
    x = [s.split() for s in x]
    return x


def evaluation(outputs, labels):
    # outputs为预测值，是一个位于0——1的概率
    # labels为标签，真实值，是0或者1
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5]  = 0
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy


# 定义词嵌入word2vec
def train_word2vec(x):
    model = Word2Vec(x, vector_size=250, window=5, min_count=5, workers=12, epochs=15, sg=1)
    return model


train_x, train_label = read_training_data()
test_x = read_testing_data()
model = train_word2vec(train_x + test_x)
model.save(w2vmodel_path)  # 保存word2vec模型


# 数据预处理类
class Preprocess():
    def __init__(self, sentences, sentence_len, w2v_path):
        self.sentences = sentences  # 句子
        self.sentence_len = sentence_len  # 句子的固定长度
        self.w2v_path = w2v_path  # w2v模型的存储地址
        self.word2index = {}  # 字典，返回目标词的下标
        self.index2word = []  # 列表，返回目标下标的词
        self.embedding_matrix = []  # 列表，返回目标下标的词的词向量

    # 将w2v的模型导入进来
    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size  # 词向量的长度

    # 添加词的词向量到词典里
    def add_embedding(self, word):
        # 这里的 word 只会是 "<PAD>" 或 "<UNK>"
        # 把一个随机生成的表征向量 vector 作为 "<PAD>" 或 "<UNK>" 的嵌入
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        # 这个新词的下标就在词典的最后一个
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    # 获取训练好的word2vec模型
    def make_embedding(self, load=True):
        print("get embedding...")
        if load:
            print("loading word to vetor model...")
            self.get_w2v_model()  # 导入w2v模型
        else:
            raise NotImplementedError

        # 遍历嵌入后的单词，并且将其导入进词典里
        # 新版gesnim里的word2vec中将mv.vocab变换了写法
        # 详见https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4里新版的特性
        for i, word in enumerate(self.embedding.wv.index_to_key):  # 新版gesnim里的word2vec中将mv.vocab变换了写法
            print('get words #{}'.format(i + 1), end='\r')
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.embedding_matrix.append((self.embedding.wv[word]))
        # 把词典的matrix形式转换为tensor形式，方便之后输入训练
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        print("get words: {}".format(len(self.embedding_matrix)))
        # 将<PAD>和<UNK>加入embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sentence(self, sentence):  # 此处senten的单词是已经转换成词典的index形式了的
        # 将每一条句子变成相同的大小
        # 若是超过了规定的长度，则截取
        if len(sentence) > self.sentence_len:
            sentence = sentence[:self.sentence_len]
        # 若没有达到，则在后面补上<PAD>单词
        else:
            pad_len = self.sentence_len - len(sentence)
            for k in range(pad_len):
                sentence.append(self.word2index["<PAD>"])  # 加上“<PAD>”在词典中的位置（index）
        assert len(sentence) == self.sentence_len
        return sentence

    def sentence_word2index(self):
        sentence_list = []  # index句子的收录列表，一行表示已经转换为单词index形式的句子
        for i, sen in enumerate(self.sentences):
            sentence_index = []  # 一个句子中单词位置列表
            for word in sen:
                if (word in self.word2index.keys()):  # 如果单词在词典里有收录
                    sentence_index.append(self.word2index[word])  # 将单词在词典的index加入进index句子中
                else:  # 如果没有收录进词典
                    sentence_index.append(self.word2index["<UNK>"])  # 用<UNK>来表示
            # 将句子变成一样的长度
            sentence_index = self.pad_sentence(sentence_index)
            # 将转换完成的句子收录进index句子列表
            sentence_list.append(sentence_index)
        return torch.LongTensor(sentence_list)

    def labels_tensor(self, labels):
        # 把label转换为tensor
        labels = [int(label) for label in labels]
        return torch.LongTensor(labels)


class myDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class BiLSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(BiLSTM_Net, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)
        # 是否将 embedding 固定住，如果 fix_embedding 为 False，在训练过程中，embedding 也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        # 分类器全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, 64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一层的 hidden state 丢到分类器中
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


def training(batch_size, n_epoch, lr, train, valid, model, device):
    # 输出模型的总参数数量，可训练的参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    loss = nn.BCELoss()  # 损失函数为二元交叉损失函数
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer设置为Ada,学习率提前设置为lr
    total_loss, total_acc, best_acc = 0, 0, 0

    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0

        # training
        model.train()
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            batch_loss = loss(outputs, labels)  # 计算此时数据的loss
            batch_loss.backward()  # 计算loss的梯度
            optimizer.step()

            accuracy = evaluation(outputs, labels)
            total_acc += (accuracy/batch_size)
            total_loss += batch_loss.item()
        print('Epoch | {}/{}'.format(epoch + 1, n_epoch))
        print('Train | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        # validation
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0

            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)

                outputs = model(inputs)
                outputs = outputs.squeeze()
                batch_loss = loss(outputs, labels)
                accuracy = evaluation(outputs, labels)
                total_acc += (accuracy / batch_size)
                total_loss += batch_loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            if total_acc > best_acc:
                # 如果 validation 的结果优于之前所有的結果，就把当下的模型保存下来，用于之后的testing
                best_acc = total_acc
                torch.save(model, finalmodel_path)
        print('-----------------------------------------------')


device = torch.device("cpu")  # 目前穷逼的我只能用cpu来跑


sen_len = 150  # 句子固定长度
fix_embedding = True
batch_size = 64
epoch = 15
lr = 0.001
w2v_path = w2vmodel_path  # 词典模型的地址

print("loadin data ...")
train_x, y = read_training_data()

preprocess =Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2index()
y = preprocess.labels_tensor(y)

# 定义模型
model = BiLSTM_Net(embedding, embedding_dim=250, hidden_dim=100, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)

# 把train——data分为training_data和validation_data两部分

x_train, x_val, y_train, y_val = train_test_split(train_x, y, test_size=0.1, random_state=1, stratify=y)
print('Train | Len:{} \nValid | Len:{}'.format(len(y_train), len(y_val)))

train_dataset = myDataset(x=x_train, y=y_train)
val_dataset = myDataset(x=x_val, y=y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

training(batch_size=batch_size, n_epoch=epoch, lr=lr, train=train_loader, valid=val_loader, model=model, device=device)


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            ret_output += outputs.int().tolist()
        return ret_output


# 开始test
print("loading testdata ...")
test_x = read_testing_data()
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2index()
test_dataset = myDataset(x=test_x, y=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("\nload model ...")
model = torch.load(finalmodel_path)
outputs = testing(batch_size=batch_size, test_loader=test_loader, model=model, device=device)

with open("work.txt", "w") as f:
    for line in outputs:
        f.write(str(line)+"\n")
f.close()








