from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

class ClassicalModelSuite:
    def __init__(self):
        self.nb = MultinomialNB()
        self.svm = LinearSVC(random_state=42, max_iter=2000)

    def train_nb(self, X_train, y_train):
        print("训练朴素贝叶斯模型...")
        self.nb.fit(X_train, y_train)

    def train_svm(self, X_train, y_train):
        print("训练SVM模型...")
        self.svm.fit(X_train, y_train)

    def predict_nb(self, X_test):
        return self.nb.predict(X_test)

    def predict_svm(self, X_test):
        return self.svm.predict(X_test)
