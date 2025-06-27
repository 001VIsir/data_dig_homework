from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

class VectorizerFactory:
    def __init__(self, max_features=5000, w2v_vector_size=100):
        self.max_features = max_features
        self.w2v_vector_size = w2v_vector_size
        self.bow_vectorizer = CountVectorizer(max_features=self.max_features)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.w2v_model = None
        self.word_to_idx = {}

    def fit_transform_bow(self, texts):
        print("使用词袋模型进行向量化...")
        return self.bow_vectorizer.fit_transform(texts)

    def transform_bow(self, texts):
        return self.bow_vectorizer.transform(texts)

    def fit_transform_tfidf(self, texts):
        print("使用TF-IDF模型进行向量化...")
        return self.tfidf_vectorizer.fit_transform(texts)

    def transform_tfidf(self, texts):
        return self.tfidf_vectorizer.transform(texts)

    def train_word2vec(self, texts):
        print("训练Word2Vec模型...")
        sentences = [text.split() for text in texts]
        self.w2v_model = Word2Vec(sentences, vector_size=self.w2v_vector_size, window=5, min_count=1, workers=4)
        # 创建词汇表索引，0号位置留给<pad>
        self.word_to_idx = {word: i + 1 for i, word in enumerate(self.w2v_model.wv.index_to_key)}
        self.word_to_idx['<pad>'] = 0

    def get_w2v_embedding_matrix(self):
        """创建可用于Embedding层的预训练权重矩阵"""
        vocab_size = len(self.word_to_idx)
        embedding_matrix = np.zeros((vocab_size, self.w2v_vector_size))
        for word, i in self.word_to_idx.items():
            if word in self.w2v_model.wv:
                embedding_matrix[i] = self.w2v_model.wv[word]
        return embedding_matrix

    def texts_to_sequences(self, texts, max_len=100):
        """将文本转换为定长的索引序列"""
        sequences = []
        for text in texts:
            # 使用get(word, 0)来处理未登录词，虽然我们这里词汇表是全的，但这是个好习惯
            seq = [self.word_to_idx.get(word, 0) for word in text.split()]
            sequences.append(seq)
        
        # 对序列进行填充或截断
        padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            padded_sequences[i, :length] = seq[:length]
            
        return padded_sequences

    def transform_word2vec(self, texts):
        if self.w2v_model is None:
            raise Exception("Word2Vec模型未训练。")
        
        features = []
        for text in texts:
            words = text.split()
            word_vectors = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
            if word_vectors:
                features.append(np.mean(word_vectors, axis=0))
            else:
                features.append(np.zeros(self.w2v_vector_size))
        return np.array(features)
