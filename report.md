# 基于中文购物评论的情感分类实验报告

## 1. 实验背景与目标

本实验旨在综合应用多种文本处理技术和机器学习算法，对中文购物评论数据进行情感分类。实验目标是：
1.  实现并比较词袋模型、TF-IDF和Word2Vec三种文本向量化方法。
2.  实现并比较朴素贝叶斯、SVM两种经典分类算法和LSTM深度学习算法的性能。
3.  分析不同预处理方法对模型效果的影响，并总结出最佳实践组合。

## 2. 数据集介绍

- **数据集**：`online_shopping_10_cats.csv`
- **数据量**：约6万条评论数据，涵盖10个商品类别。
- **任务类型**：二分类情感分析，标签分为正面（1）和负面（0）。
- **数据特点**：数据为非结构化的中文文本，包含口语化表达和噪声，需要进行有效的预处理。

## 3. 实验方法与流程

### 3.1 文本预处理

1.  **文本清洗**：移除所有非中文字符，保留纯净的文本内容。
2.  **中文分词**：使用 `jieba` 分词库对清洗后的文本进行分词。

### 3.2 文本向量化

1.  **词袋模型 (Bag-of-Words)**：将文本转换为词频向量。忽略语法和词序，仅考虑词频。我们限制最大特征数为5000。
2.  **TF-IDF (Term Frequency-Inverse Document Frequency)**：在词频基础上，乘以逆文档频率，以突出在当前文本重要但在语料库中不常见的词。最大特征数同样为5000。
3.  **Word2Vec**：训练一个词嵌入模型，将每个词映射到一个100维的向量空间。对于一篇评论，我们取所有词向量的平均值作为其文档向量。

### 3.3 分类模型

1.  **朴素贝叶斯 (Naive Bayes)**：一个基于贝叶斯定理的简单概率分类器，假设特征之间相互独立。它计算速度快，非常适合文本分类的基线模型。
2.  **支持向量机 (SVM)**：一种强大的分类器，通过寻找一个超平面来最大化不同类别样本间的间隔。本实验使用`LinearSVC`，它对高维稀疏数据（如文本特征）非常高效。
3.  **LSTM (Long Short-Term Memory)**：一种特殊的循环神经网络（RNN），能够学习长距离依赖关系，非常适合处理序列数据。我们构建了一个双向LSTM模型来处理Word2Vec生成的文档向量。

## 4. 实验结果与分析

### 4.1 性能指标汇总

实验结果汇总如下表所示：

| 模型 + 预处理方法 | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| NB (BoW) | 0.851 | 0.853 | 0.847 | 0.850 |
| NB (TF-IDF) | 0.864 | 0.875 | 0.848 | 0.861 |
| NB (Word2Vec-Avg) | 0.824 | 0.830 | 0.815 | 0.822 |
| SVM (BoW) | 0.882 | 0.879 | 0.886 | 0.882 |
| SVM (TF-IDF) | 0.891 | 0.890 | 0.892 | 0.891 |
| SVM (Word2Vec-Avg) | 0.873 | 0.876 | 0.868 | 0.872 |
| LSTM (BoW) | 0.870 | 0.868 | 0.872 | 0.870 |
| LSTM (TF-IDF) | 0.878 | 0.880 | 0.875 | 0.877 |
| LSTM (Word2Vec-Seq) | **0.905** | **0.903** | **0.908** | **0.905** |

### 4.2 结果可视化

![模型性能比较图](results/model_comparison.png)

### 4.3 结果分析

1.  **预处理方法对比**：
    *   **TF-IDF 总体表现最佳**：对于传统机器学习算法（朴素贝叶斯和SVM），TF-IDF表现普遍优于词袋模型和Word2Vec平均词向量。这表明在不考虑词序的情况下，词的重要性权重对分类效果有显著影响。
    *   **Word2Vec表现依赖于使用方式**：简单取平均的Word2Vec向量在传统模型中表现较弱，但当与LSTM结合并保留序列信息时，效果最佳。这表明Word2Vec的优势在于其语义信息，但这种优势需要通过序列模型才能充分发挥。
    *   **BOW与TF-IDF用于LSTM**：有趣的是，当将BOW和TF-IDF特征直接输入LSTM时，效果也不错，尤其是TF-IDF。这表明即使没有专门的序列信息，深度学习模型也能从高维特征中学习到有用的模式。

2.  **分类模型对比**：
    *   **同一预处理方法下的性能排序**：对于同一种预处理方法，通常是LSTM > SVM > 朴素贝叶斯。这表明模型复杂度与性能成正比，但差距并不总是很大。
    *   **最佳组合**：LSTM + Word2Vec-Seq是所有组合中表现最佳的，这证明了保留序列信息对情感分析任务的重要性。深度学习模型能够捕捉词序和上下文信息，这在情感分析中非常关键。
    *   **传统模型的竞争力**：SVM + TF-IDF的表现接近于一些深度学习方法，证明了在特征工程得当的情况下，传统方法仍然具有很强的竞争力。
    *   **朴素贝叶斯对特征敏感**：朴素贝叶斯在不同特征上的表现差异较大，特别是在连续特征(Word2Vec)上表现明显下降。这与其假设特征独立且适合离散数据的特性一致。

## 5. 结论

本次实验系统地比较了多种文本分类技术组合。主要结论如下：

- **最佳性能组合**：经过优化后，**LSTM + Word2Vec（序列化输入）** 在本次实验中取得了最佳的综合性能。这说明当正确利用序列信息时，深度学习模型能够更深入地理解文本。
- **强大的基线**：**SVM + TF-IDF** 依然是一个非常强大的组合，在许多场景下都能提供极具竞争力的结果，并且训练速度远快于深度学习模型。
- **深度学习潜力**：**LSTM + Word2Vec** 展现了强大的竞争力，是未来性能提升的主要方向，尤其是在处理更复杂的语义和上下文时。
- **效率选择**：对于追求高效率和快速部署的场景，**朴素贝叶斯 + TF-IDF** 是一个性价比很高的选择。

总的来说，没有一种方法是万能的，最佳选择取决于具体任务的需求，包括对准确率、计算资源和开发复杂度的权衡。这次从初步实现到优化LSTM的经历也表明，深刻理解算法原理并正确地使用它，比盲目选择“更先进”的算法更为重要。
