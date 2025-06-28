from data.data_loader import ChineseReviewLoader
from preprocessing.vectorizers import VectorizerFactory
from models.classical_models import ClassicalModelSuite
from models.deep_models import LSTMTrainer
from evaluation.metrics import calculate_metrics, plot_and_save_results
import pandas as pd
import os
import numpy as np

def run_experiment():
    # 1. 加载数据
    # 注意：请确保数据集位于 'data' 文件夹下
    data_path = 'online_shopping_10_cats\online_shopping_10_cats.csv'
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
    
    # 文本平均词向量 (用于传统模型)
    X_train_w2v_avg = vectorizer.transform_word2vec(X_train)
    X_test_w2v_avg = vectorizer.transform_word2vec(X_test)

    # 3. 训练和评估
    results = {}
    # 创建一个字典来存储所有预测结果
    predictions_dict = {
        'text': X_test,
        'true_label': y_test
    }
    models = ClassicalModelSuite()

    # 3.1 朴素贝叶斯
    print("\n--- 朴素贝叶斯模型实验 ---")
    # BoW特征
    models.train_nb(X_train_bow, y_train)
    y_pred = models.predict_nb(X_test_bow)
    results['NB (BoW)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['NB (BoW)'] = y_pred

    # TF-IDF特征
    models.train_nb(X_train_tfidf, y_train)
    y_pred = models.predict_nb(X_test_tfidf)
    results['NB (TF-IDF)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['NB (TF-IDF)'] = y_pred
    
    # Word2Vec平均词向量特征
    print("训练基于Word2Vec的朴素贝叶斯模型...")
    models.train_nb(X_train_w2v_avg, y_train)
    y_pred = models.predict_nb(X_test_w2v_avg)
    results['NB (Word2Vec-Avg)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['NB (Word2Vec-Avg)'] = y_pred

    # 3.2 SVM
    print("\n--- SVM模型实验 ---")
    # BoW特征
    models.train_svm(X_train_bow, y_train)
    y_pred = models.predict_svm(X_test_bow)
    results['SVM (BoW)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['SVM (BoW)'] = y_pred

    # TF-IDF特征
    models.train_svm(X_train_tfidf, y_train)
    y_pred = models.predict_svm(X_test_tfidf)
    results['SVM (TF-IDF)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['SVM (TF-IDF)'] = y_pred
    
    # Word2Vec平均词向量特征
    print("训练基于Word2Vec的SVM模型...")
    models.train_svm(X_train_w2v_avg, y_train)
    y_pred = models.predict_svm(X_test_w2v_avg)
    results['SVM (Word2Vec-Avg)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['SVM (Word2Vec-Avg)'] = y_pred
    
    # 3.3 LSTM模型实验
    print("\n--- LSTM模型实验 ---")
    
    # 3.3.1 基于BoW的LSTM
    print("准备基于BoW的LSTM...")
    # 将稀疏矩阵转换为密集数组
    X_train_bow_dense = X_train_bow.toarray()
    X_test_bow_dense = X_test_bow.toarray()
    
    bow_input_dim = X_train_bow_dense.shape[1]
    lstm_bow = LSTMTrainer(vocab_size=bow_input_dim, embedding_dim=bow_input_dim)
    print("训练基于BoW的LSTM模型...")
    lstm_bow.train(X_train_bow_dense, y_train, epochs=10)
    y_pred = lstm_bow.predict(X_test_bow_dense)
    results['LSTM (BoW)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['LSTM (BoW)'] = y_pred
    
    # 3.3.2 基于TF-IDF的LSTM
    print("准备基于TF-IDF的LSTM...")
    # 将稀疏矩阵转换为密集数组
    X_train_tfidf_dense = X_train_tfidf.toarray()
    X_test_tfidf_dense = X_test_tfidf.toarray()
    
    tfidf_input_dim = X_train_tfidf_dense.shape[1]
    lstm_tfidf = LSTMTrainer(vocab_size=tfidf_input_dim, embedding_dim=tfidf_input_dim)
    print("训练基于TF-IDF的LSTM模型...")
    lstm_tfidf.train(X_train_tfidf_dense, y_train, epochs=10)
    y_pred = lstm_tfidf.predict(X_test_tfidf_dense)
    results['LSTM (TF-IDF)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['LSTM (TF-IDF)'] = y_pred
    
    # 3.3.3 优化的LSTM (基于Word2Vec序列)
    print("\n--- 基于Word2Vec序列的优化LSTM实验 ---")
    # 将文本转换为索引序列
    MAX_LEN = 100  # 定义评论的最大长度
    X_train_seq = vectorizer.texts_to_sequences(X_train, max_len=MAX_LEN)
    X_test_seq = vectorizer.texts_to_sequences(X_test, max_len=MAX_LEN)

    # 获取词汇表大小和预训练权重
    vocab_size = len(vectorizer.word_to_idx)
    embedding_matrix = vectorizer.get_w2v_embedding_matrix()
    embedding_dim = vectorizer.w2v_vector_size

    # 初始化并训练模型
    lstm_trainer = LSTMTrainer(vocab_size=vocab_size, embedding_dim=embedding_dim)
    lstm_trainer.load_pretrained_embeddings(embedding_matrix)  # 加载预训练权重
    lstm_trainer.train(X_train_seq, y_train, epochs=20)
    
    # 评估
    y_pred = lstm_trainer.predict(X_test_seq)
    results['LSTM (Word2Vec-Seq)'] = calculate_metrics(y_test, y_pred)
    predictions_dict['LSTM (Word2Vec-Seq)'] = y_pred

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
    
    # 将标签从数字映射到文本，方便阅读
    label_map = {0: '负面', 1: '正面'}
    # 获取所有需要映射的列（除了'text'列）
    label_columns = [col for col in predictions_df.columns if col != 'text']
    for col in label_columns:
        predictions_df[col] = predictions_df[col].map(label_map)
        
    # 使用 utf-8-sig 编码以确保在Excel中正确显示中文
    predictions_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n预测对比文件已保存到: {comparison_path}")


if __name__ == '__main__':
    run_experiment()
