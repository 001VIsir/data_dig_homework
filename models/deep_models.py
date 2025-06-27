import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128, output_dim=2, n_layers=2, dropout=0.5, padding_idx=0):
        super().__init__()
        # 1. Embedding层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # 2. LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        
        # 3. 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # 双向所以*2
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_indices):
        # text_indices = [batch size, sent len]
        
        # embedded = [batch size, sent len, emb dim]
        embedded = self.embedding(text_indices)
        
        # packed_output, (hidden, cell)
        # hidden = [num layers * num directions, batch size, hid dim]
        _, (hidden, cell) = self.lstm(embedded)
        
        # 拼接双向LSTM的最后一个隐藏层状态
        # hidden = [batch size, hid dim * num directions]
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        return self.fc(hidden)

class LSTMTrainer:
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128, n_layers=2, dropout=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, n_layers=n_layers, dropout=dropout).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def load_pretrained_embeddings(self, embedding_matrix):
        """加载预训练的词向量权重"""
        self.model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # 可选：冻结Embedding层，使其在训练中不更新
        # self.model.embedding.weight.requires_grad = False

    # 这里的 epochs=5 是默认值
    def train(self, X_train, y_train, epochs=5, batch_size=64):
        print(f"在 {self.device} 上训练LSTM模型...")
        dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
        loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for texts, labels in loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(texts) # .squeeze(1) is no longer needed
                loss = self.criterion(predictions, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f'Epoch {epoch+1:02} | Loss: {epoch_loss/len(loader):.4f}')

    def predict(self, X_test, batch_size=64):
        self.model.eval()
        dataset = TensorDataset(torch.LongTensor(X_test))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        all_preds = []
        with torch.no_grad():
            for texts in loader:
                texts = texts[0].to(self.device)
                predictions = self.model(texts) # .squeeze(1) is no longer needed
                _, predicted_labels = torch.max(predictions, 1)
                all_preds.extend(predicted_labels.cpu().numpy())
        return np.array(all_preds)
