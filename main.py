from data.data_loader import ChineseReviewLoader
from preprocessing.vectorizers import VectorizerFactory
from models.classical_models import ClassicalModelSuite
from models.deep_models import LSTMTrainer
from evaluation.metrics import calculate_metrics, plot_and_save_results
import pandas as pd
import os

def run_experiment():
    # 1. 加载数据
    # 注意：请确保数据集位于 'data' 文件夹下
    data_path = 'D:\PycharmProjects\PythonProject5\data_dig_homework\online_shopping_10_cats\online_shopping_10_cats.csv'
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        # 尝试修复用户提供的可能不正确的路径
        alt_path = 'online_shopping_10_cats.csv'
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            return

    loader = ChineseReviewLoader(file_path=data_path)
    X_train, X_test, y_train, y_test = loader.load_and_process()

    # 2. 向量化
    vectorizer = VectorizerFactory(max_features=5000, w2v_vector_size=100)
    
    # 词袋
    X_train_bow = vectorizer.fit_transform_bow(X_train)
    X_test_bow = vectorizer.transform_bow(X_test)
    
    # TF-IDF
    X_train_tfidf = vectorizer.fit_transform_tfidf(X_train)
    X_test_tfidf = vectorizer.transform_tfidf(X_test)
    
    # Word2Vec
    vectorizer.train_word2vec(X_train)
    # 原始的基于平均向量的方法（保留用于对比或移除）
    # X_train_w2v_avg = vectorizer.transform_word2vec(X_train)
    # X_test_w2v_avg = vectorizer.transform_word2vec(X_test)

    # 3. 训练和评估
    results = {}
    # 创建一个字典来存储所有预测结果
    predictions_dict = {
        'text': X_test,
        'true_label': y_test
    }
    models = ClassicalModelSuite()

    # 朴素贝叶斯
    models.train_nb(X_train_bow, y_train)
    y_pred = models.predict_nb(X_test_bow)
    results['NB (BoW)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['NB (BoW)'] = y_pred

    models.train_nb(X_train_tfidf, y_train)
    y_pred = models.predict_nb(X_test_tfidf)
    results['NB (TF-IDF)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['NB (TF-IDF)'] = y_pred

    # SVM
    models.train_svm(X_train_bow, y_train)
    y_pred = models.predict_svm(X_test_bow)
    results['SVM (BoW)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['SVM (BoW)'] = y_pred

    models.train_svm(X_train_tfidf, y_train)
    y_pred = models.predict_svm(X_test_tfidf)
    results['SVM (TF-IDF)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['SVM (TF-IDF)'] = y_pred
    
    # --- 优化的 LSTM 流程 ---
    print("\n--- 开始优化后的 LSTM 实验 ---")
    # 1. 将文本转换为索引序列
    MAX_LEN = 100 # 定义评论的最大长度
    X_train_seq = vectorizer.texts_to_sequences(X_train, max_len=MAX_LEN)
    X_test_seq = vectorizer.texts_to_sequences(X_test, max_len=MAX_LEN)

    # 2. 获取词汇表大小和预训练权重
    vocab_size = len(vectorizer.word_to_idx)
    embedding_matrix = vectorizer.get_w2v_embedding_matrix()
    embedding_dim = vectorizer.w2v_vector_size

    # 3. 初始化并训练模型
    lstm_trainer = LSTMTrainer(vocab_size=vocab_size, embedding_dim=embedding_dim)
    lstm_trainer.load_pretrained_embeddings(embedding_matrix) # 加载预训练权重
    lstm_trainer.train(X_train_seq, y_train, epochs=20)
    
    # 4. 评估
    y_pred = lstm_trainer.predict(X_test_seq)
    results['Optimized LSTM (Word2Vec)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['Optimized LSTM (Word2Vec)'] = y_pred

    # 4. 结果展示
    results_df = pd.DataFrame(results).T
    print("\n--- 实验结果汇总 ---")
    print(results_df)
    
    plot_and_save_results(results_df)

    # 5. 保存预测对比文件
    predictions_df = pd.DataFrame(predictions_dict)
    # 确保 'results' 目录存在
    os.makedirs('results', exist_ok=True)
    comparison_path = 'results/prediction_comparison.csv'
    # 使用 utf-8-sig 编码以确保在Excel中正确显示中文
    predictions_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n预测对比文件已保存到: {comparison_path}")


if __name__ == '__main__':
    run_experiment()
