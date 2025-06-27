import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
import re

class ChineseReviewLoader:
    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        # 可选：加载停用词表，此处为简单示例
        # return set([line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()])
        return set()

    @staticmethod
    def clean_and_segment(text):
        """清洗并分词"""
        if not isinstance(text, str):
            return ""
        # 1. 去除非中文字符
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        # 2. 分词
        words = jieba.lcut(text)
        # 3. 去除停用词（如果需要）和空格
        # words = [w for w in words if w not in self.stopwords and w.strip()]
        words = [w for w in words if w.strip()]
        return ' '.join(words)

    def load_and_process(self):
        """加载数据、预处理并划分"""
        print("正在加载和预处理数据...")
        df = pd.read_csv(self.file_path)
        
        # 删除缺失值
        df.dropna(subset=['review'], inplace=True)
        
        # 确保标签是整数
        df['label'] = df['label'].astype(int)
        
        # 应用预处理
        df['processed_review'] = df['review'].apply(self.clean_and_segment)
        
        # 划分数据集
        X = df['processed_review'].values
        y = df['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"数据加载完成。训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
        return X_train, X_test, y_train, y_test
