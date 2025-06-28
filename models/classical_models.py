from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
import numpy as np

class ClassicalModelSuite:
    def __init__(self):
        self.multinb = MultinomialNB()  # 多项式朴素贝叶斯，适用于BoW和TF-IDF
        self.gaussnb = GaussianNB()     # 高斯朴素贝叶斯，适用于Word2Vec
        self.svm = LinearSVC(random_state=42, max_iter=2000)

    def train_nb(self, X_train, y_train):
        print("训练朴素贝叶斯模型...")
        
        # 检查输入数据类型和是否包含负值
        if isinstance(X_train, np.ndarray) and np.any(X_train < 0):
            # 如果包含负值，使用高斯朴素贝叶斯
            print("检测到负值，使用高斯朴素贝叶斯")
            self.gaussnb.fit(X_train, y_train)
            self.current_nb = 'gaussian'
        else:
            # 否则使用多项式朴素贝叶斯
            # 如果输入是稀疏矩阵，不需要检查负值，因为BoW和TF-IDF通常是非负的
            try:
                self.multinb.fit(X_train, y_train)
                self.current_nb = 'multinomial'
            except ValueError:
                # 如果仍然失败（可能是密集数组中有负值），回退到高斯模型
                print("MultinomialNB失败，回退到高斯朴素贝叶斯")
                self.gaussnb.fit(X_train, y_train)
                self.current_nb = 'gaussian'

    def train_svm(self, X_train, y_train):
        print("训练SVM模型...")
        self.svm.fit(X_train, y_train)

    def predict_nb(self, X_test):
        if hasattr(self, 'current_nb') and self.current_nb == 'gaussian':
            return self.gaussnb.predict(X_test)
        else:
            try:
                return self.multinb.predict(X_test)
            except ValueError:
                # 如果预测失败，尝试使用高斯模型（如果已经训练过）
                if hasattr(self.gaussnb, 'classes_'):
                    return self.gaussnb.predict(X_test)
                else:
                    raise ValueError("无法预测：多项式模型失败且高斯模型未训练")

    def predict_svm(self, X_test):
        return self.svm.predict(X_test)
