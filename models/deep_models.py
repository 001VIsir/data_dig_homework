import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128, output_dim=2, n_layers=2, dropout=0.5, padding_idx=0):
        super().__init__()
        
        # 检查是否使用序列数据或特征向量
        self.is_bow_or_tfidf = (vocab_size == embedding_dim)
        
        if self.is_bow_or_tfidf:
            # 对于BoW或TF-IDF，我们不使用嵌入层，直接使用特征向量
            self.fc_input = nn.Linear(embedding_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, 
                                bidirectional=True, dropout=dropout, batch_first=True)
        else:
            # 对于Word2Vec序列，我们使用嵌入层
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                                bidirectional=True, dropout=dropout, batch_first=True)
        
        # 最后的全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.is_bow_or_tfidf:
            # 如果输入是BoW或TF-IDF特征
            # x形状: [batch_size, features]
            
            # 将特征转换为隐藏维度
            x = self.fc_input(x)  # [batch_size, hidden_dim]
            
            # 添加序列维度以输入LSTM
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 通过LSTM
            _, (hidden, _) = self.lstm(x)
        else:
            # 如果输入是词索引序列
            # x形状: [batch_size, seq_len]
            
            # 通过嵌入层
            embedded = self.embedding(x)  # [batch_size, seq_len, emb_dim]
            
            # 通过LSTM
            _, (hidden, _) = self.lstm(embedded)
        
        # 拼接双向LSTM的隐藏状态
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        # 全连接分类层
        return self.fc(hidden)

class LSTMTrainer:
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128, n_layers=2, dropout=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, n_layers=n_layers, dropout=dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.is_bow_or_tfidf = (vocab_size == embedding_dim)

    def load_pretrained_embeddings(self, embedding_matrix):
        """加载预训练的词向量权重"""
        if not self.is_bow_or_tfidf:
            self.model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            # 可选：冻结Embedding层，使其在训练中不更新
            # self.model.embedding.weight.requires_grad = False

    def train(self, X_train, y_train, epochs=5, batch_size=64):
        print(f"在 {self.device} 上训练LSTM模型...")
        
        # 根据输入类型选择合适的张量类型
        if self.is_bow_or_tfidf:
            # BoW或TF-IDF特征为浮点型
            X_tensor = torch.FloatTensor(X_train)
        else:
            # 词索引序列为整型
            X_tensor = torch.LongTensor(X_train)
            
        dataset = TensorDataset(X_tensor, torch.LongTensor(y_train))
        loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=2)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for texts, labels in loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(texts)
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f'Epoch {epoch+1:02} | Loss: {epoch_loss/len(loader):.4f}')

    def predict(self, X_test, batch_size=64):
        self.model.eval()
        
        # 根据输入类型选择合适的张量类型
        if self.is_bow_or_tfidf:
            # BoW或TF-IDF特征为浮点型
            X_tensor = torch.FloatTensor(X_test)
        else:
            # 词索引序列为整型
            X_tensor = torch.LongTensor(X_test)
            
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        all_preds = []
        with torch.no_grad():
            for texts in loader:
                texts = texts[0].to(self.device)
                predictions = self.model(texts)
                _, predicted_labels = torch.max(predictions, 1)
                all_preds.extend(predicted_labels.cpu().numpy())
        return np.array(all_preds)
