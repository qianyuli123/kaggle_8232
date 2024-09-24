import torch
import torch.nn as nn
import pandas as pd
from dataload import DataLoader

# 自定义文本分词函数
def custom_tokenize(text):
    import re
    return re.findall(r'\b\w+\b', text.lower())

# 构建词汇表
def build_vocab(texts):
    word_count = {}
    for text in texts:
        tokens = custom_tokenize(text)
        for token in tokens:
            if token in word_count:
                word_count[token] += 1
            else:
                word_count[token] = 1
    
    # 添加特殊标记
    word_count['<pad>'] = 1e9  # PAD标记
    word_count['<unk>'] = 1e9  # OOV标记
    
    # 构建词汇表
    vocab = {word: idx for idx, (word, _) in enumerate(sorted(word_count.items(), key=lambda x: -x[1]))}
    return vocab

# 定义一个简单的RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return out[:, -1, :]

# 定义Encoder类
class Encoder:
    def __init__(self, train_df: pd.DataFrame, mapping_df: pd.DataFrame, hidden_size=128, embedding_dim=64, device='cuda'):
        self.train_df = train_df
        self.mapping_df = mapping_df
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # 构建从 MisconceptionId 到 MisconceptionName 的映射表
        self.misconception_map = dict(zip(self.mapping_df['MisconceptionId'], self.mapping_df['MisconceptionName']))
        
        # 构建词汇表
        texts = train_df['QuestionText'].tolist()
        self.vocab = build_vocab(texts)
        self.vocab_size = len(self.vocab)
        
        # 初始化RNN模型
        self.feature_rnn = RNN(self.embedding_dim, self.hidden_size).to(device)
        self.label_rnn = RNN(self.embedding_dim, self.hidden_size).to(device)
        
        # 初始化嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(device)
        
    def tokenize_and_encode(self, text):
        tokens = custom_tokenize(text)
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(indices, dtype=torch.long).to(self.device)
    
    def encode_features(self, row):
        """
        将单行数据中的 ConstructName、SubjectName、QuestionText、AnswerAText 拼接起来并编码成向量。
        
        参数:
        - row: DataFrame中的单行记录。
        
        返回:
        - 编码后的特征向量。
        """
        concatenated_text = f"{row['ConstructName']} {row['SubjectName']} {row['QuestionText']} {row['AnswerText']}"
        tokenized_text = self.tokenize_and_encode(concatenated_text)
        embedded_text = self.embedding(tokenized_text.unsqueeze(0))
        feature_vector = self.feature_rnn(embedded_text)
        return feature_vector
    
    def encode_labels(self, row):
        """
        将单行数据中的 MisconceptionName 编码成向量。
        
        参数:
        - row: DataFrame中的单行记录。
        
        返回:
        - 编码后的标签向量。
        """
        misconception_name = self.misconception_map[row['MisconceptionId']]
        tokenized_text = self.tokenize_and_encode(misconception_name)
        embedded_text = self.embedding(tokenized_text.unsqueeze(0))
        label_vector = self.label_rnn(embedded_text)
        return label_vector
    
    def similarity_loss(self, feature_vector, label_vector):
        """
        计算特征向量和标签向量之间的相似性损失。
        
        参数:
        - feature_vector: 特征向量。
        - label_vector: 标签向量。
        
        返回:
        - 相似性损失。
        """
        similarity = torch.cosine_similarity(feature_vector, label_vector, dim=1)
        loss = -similarity.mean()  # 目标是最小化负相似性，即最大化相似性
        return loss
    
    def train(self, epochs=10, learning_rate=0.001):
        """
        训练模型。
        
        参数:
        - epochs: 训练轮数。
        - learning_rate: 学习率。
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CosineEmbeddingLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for index, row in self.train_df.iterrows():
                feature_vector = self.encode_features(row)
                label_vector = self.encode_labels(row)
                
                optimizer.zero_grad()
                loss = self.similarity_loss(feature_vector, label_vector)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_df)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    def save_model(self, filepath):
        """
        保存模型状态字典到指定路径。
        
        参数:
        - filepath: 文件路径。
        """
        torch.save({
            'embedding_state_dict': self.embedding.state_dict(),
            'feature_rnn_state_dict': self.feature_rnn.state_dict(),
            'label_rnn_state_dict': self.label_rnn.state_dict(),
            'vocab': self.vocab,
            'misconception_map': self.misconception_map
        }, filepath)
    
    def load_model(self, filepath):
        """
        加载模型状态字典。
        
        参数:
        - filepath: 文件路径。
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
        self.feature_rnn.load_state_dict(checkpoint['feature_rnn_state_dict'])
        self.label_rnn.load_state_dict(checkpoint['label_rnn_state_dict'])
        self.vocab = checkpoint['vocab']
        self.misconception_map = checkpoint['misconception_map']

    def parameters(self):
        return list(self.embedding.parameters()) + list(self.feature_rnn.parameters()) + list(self.label_rnn.parameters())