# 中文购物评论情感分类项目文档

## 1. 数据集概述

### 1.1 数据来源

本项目使用的数据集为中文电商购物评论数据集（`online_shopping_10_cats.csv`）。该数据集收集自中国主要电商平台的用户评论，涵盖10个不同的商品类别。数据集已经过初步处理，并进行了情感标注（正面/负面），适合进行二分类情感分析研究。
https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/online_shopping_10_cats

### 1.2 属性名称与属性类型

数据集包含以下字段：

| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| label | 整型 (int) | 情感标签：0表示负面评论，1表示正面评论 |
| review | 字符串 (string) | 用户评论文本内容 |
| cat | 字符串 (string) | 商品类别，包括"书籍"、"平板"、"手机"、"水果"、"洗发水"、"热水器"、"蒙牛"、"衣服"、"计算机"、"酒店"共10个类别 |

### 1.3 数据规模

- **总记录数**：约60,000条评论数据
- **类别分布**：
  - 正面评论（label=1）：约30,000条
  - 负面评论（label=0）：约30,000条
- **类别数量**：10个商品类别，每个类别的评论数量大致相当
- **文本长度**：评论长度不一，从几个字到数百字不等，平均长度约为25-30个中文字符

### 1.4 数据样例

以下是数据集中的几个样例记录：

#### 正面评论示例：

```
label: 1
review: 这本书写得很好，内容丰富，讲解透彻，很值得一读
cat: 书籍
```

```
label: 1
review: 手机质量不错，运行速度快，电池续航也很好，很满意
cat: 手机
```

#### 负面评论示例：

```
label: 0
review: 酒店卫生条件太差，床单有异味，而且服务态度也不好
cat: 酒店
```

```
label: 0
review: 衣服质量一般，而且尺码偏小，穿着不舒服，不推荐
cat: 衣服
```

注意：本项目主要关注情感分类任务，因此主要使用 `review` 文本和 `label` 标签，而暂时未利用 `cat` 类别信息。

## 2. 分析目标

本项目旨在对中文购物评论数据进行情感分析，重点关注以下几个方面：

### 2.1 主要目标

1. **情感二分类**：构建能够准确区分正面评论和负面评论的分类模型。
2. **方法比较**：系统地比较不同的文本向量化方法和分类算法的效果。
3. **最优组合**：确定最适合中文评论情感分析的预处理方法和模型组合。
4. **模型优化**：探索如何优化深度学习模型，提高其在情感分类任务中的表现。

### 2.2 研究问题

1. 不同的文本向量化方法（词袋、TF-IDF、Word2Vec）对情感分类任务的影响如何？
2. 传统机器学习模型（朴素贝叶斯、SVM）与深度学习模型（LSTM）在中文情感分析中的性能差异？
3. 不同预处理方法与分类算法的组合效果如何？例如，TF-IDF+SVM 或 Word2Vec+LSTM 是否有明显优势？
4. 如何优化模型参数和输入形式以获得最佳分类效果？

### 2.3 预期成果

1. 建立一个准确率高于85%的中文评论情感分类模型。
2. 形成一套适合中文短文本情感分析的方法论。
3. 对不同方法和模型的优缺点进行系统性总结，提供明确的应用指导。
4. 提供一个可以直观对比不同模型预测结果的分析框架。

## 3. 数据预处理及文本向量化

本节详细介绍从原始文本数据到可用于模型训练的特征向量的转换过程。由于我们处理的是中文文本，预处理步骤尤为重要，它直接影响模型的性能。

### 3.1 构建词汇表

词汇表是文本向量化的基础，它记录了语料库中所有出现的词语及其对应的索引。在本项目中，词汇表通过以下方式构建：

- **词袋模型和TF-IDF**：通过 `scikit-learn` 的 `CountVectorizer` 和 `TfidfVectorizer` 自动构建，并限制最大特征数为5000。
- **Word2Vec**：通过训练 Word2Vec 模型获得词汇表，并将其转换为词到索引的映射，以供后续深度学习模型使用。

```python
# Word2Vec词汇表构建示例
self.word_to_idx = {word: i + 1 for i, word in enumerate(self.w2v_model.wv.index_to_key)}
self.word_to_idx['<pad>'] = 0  # 添加填充标记
```

### 3.2 设置路径以及读取数据集

首先，我们需要设置数据集路径，并使用 `pandas` 读取 CSV 文件：

```python
# 数据集路径设置
data_path = 'D:\PycharmProjects\PythonProject5\data_dig_homework\online_shopping_10_cats\online_shopping_10_cats.csv'

# 使用pandas读取数据
df = pd.read_csv(data_path)

# 检查数据是否存在缺失值，并删除
df.dropna(subset=['review'], inplace=True)

# 确保标签是整数类型
df['label'] = df['label'].astype(int)
```

通过 `pandas` 读取 CSV 数据后，我们获得了包含评论文本、情感标签和商品类别的结构化数据框。

### 3.3 移除特殊字符

为了提高文本质量，我们需要移除评论中的特殊字符、标点符号和数字等噪声，只保留有意义的汉字：

```python
# 使用正则表达式移除非中文字符
text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
```

正则表达式 `r'[^\u4e00-\u9fa5]'` 匹配所有非中文字符，并将其替换为空字符串。

### 3.4 分词

中文与英文不同，没有明确的词语边界，因此需要使用分词技术将句子切分为词语序列。本项目使用 `jieba` 分词库进行中文分词：

```python
# 使用jieba进行分词
words = jieba.lcut(text)
```

分词后，每个评论文本被转换为词语列表，这是后续向量化的基础。

### 3.5 去除停用词

停用词是指那些在文本中频繁出现但对文本分类没有实质性帮助的常见词语（如"的"、"是"、"在"等）。虽然在当前实现中我们没有使用停用词表，但代码结构支持这一功能：

```python
# 如果需要去除停用词，取消下面代码的注释
# words = [w for w in words if w not in self.stopwords and w.strip()]

# 当前实现只去除空格
words = [w for w in words if w.strip()]
```

### 3.6 标签映射

在情感分析任务中，我们需要将文本标签映射为数值形式：

- 负面评论（Negative）映射为 0
- 正面评论（Positive）映射为 1

在数据集中，标签已经是整数形式，但我们确保其类型正确：

```python
# 确保标签是整数类型
df['label'] = df['label'].astype(int)
```

在输出结果时，我们再将数值标签映射回文本标签，以增强可读性：

```python
# 将数值标签映射回文本标签
label_map = {0: '负面', 1: '正面'}
for col in label_columns:
    predictions_df[col] = predictions_df[col].map(label_map)
```

### 3.7 文本向量化

文本向量化是将文本转换为数值特征向量的过程，是机器学习模型处理文本数据的必要步骤。本项目实现了三种常用的文本向量化方法：

#### 3.7.1 词袋预处理技术

词袋模型（Bag of Words, BoW）是最简单的文本向量化方法，它将每个文档表示为一个向量，向量中的每个元素对应词汇表中的一个词，其值表示该词在文档中出现的次数。

```python
# 创建词袋向量器
self.bow_vectorizer = CountVectorizer(max_features=self.max_features)

# 应用词袋模型转换文本
def fit_transform_bow(self, texts):
    print("使用词袋模型进行向量化...")
    return self.bow_vectorizer.fit_transform(texts)
```

优点：
- 实现简单，计算高效
- 保留词频信息

缺点：
- 忽略词序和语义
- 维度高，向量稀疏

#### 3.7.2 TF-IDF预处理技术

TF-IDF（Term Frequency-Inverse Document Frequency）是对词袋模型的改进，它不仅考虑词语在文档中的频率，还考虑词语在整个语料库中的稀有程度。

```python
# 创建TF-IDF向量器
self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features)

# 应用TF-IDF模型转换文本
def fit_transform_tfidf(self, texts):
    print("使用TF-IDF模型进行向量化...")
    return self.tfidf_vectorizer.fit_transform(texts)
```

优点：
- 考虑词语的重要性，降低常见词的权重
- 提高稀有但重要词的权重

缺点：
- 仍然忽略词序和语义关系
- 无法处理同义词和多义词

#### 3.7.3 Word2Vec词向量

Word2Vec是一种基于神经网络的词嵌入技术，它可以将词语映射到低维连续向量空间，使得语义相近的词在这个空间中的位置也相近。

```python
# 训练Word2Vec模型
def train_word2vec(self, texts):
    print("训练Word2Vec模型...")
    sentences = [text.split() for text in texts]
    self.w2v_model = Word2Vec(sentences, vector_size=self.w2v_vector_size, 
                             window=5, min_count=1, workers=4)
```

为了将Word2Vec与深度学习模型结合，我们实现了两种方法：

1. **平均词向量**：计算文档中所有词向量的平均值作为文档向量。
   ```python
   # 转换为平均词向量
   def transform_word2vec(self, texts):
       features = []
       for text in texts:
           words = text.split()
           word_vectors = [self.w2v_model.wv[word] for word in words 
                          if word in self.w2v_model.wv]
           if word_vectors:
               features.append(np.mean(word_vectors, axis=0))
           else:
               features.append(np.zeros(self.w2v_vector_size))
       return np.array(features)
   ```

2. **词索引序列**：将文本转换为词索引序列，并在模型中使用嵌入层。
   ```python
   # 将文本转换为词索引序列
   def texts_to_sequences(self, texts, max_len=100):
       sequences = []
       for text in texts:
           seq = [self.word_to_idx.get(word, 0) for word in text.split()]
           sequences.append(seq)
       
       # 对序列进行填充或截断
       padded_sequences = np.zeros((len(sequences), max_len), dtype=int)
       for i, seq in enumerate(sequences):
           length = min(len(seq), max_len)
           padded_sequences[i, :length] = seq[:length]
           
       return padded_sequences
   ```

优点：
- 捕捉词语的语义关系
- 降低维度，解决稀疏性问题
- 可以处理同义词和多义词

缺点：
- 需要较多的训练数据
- 计算资源需求较高
- 在简单任务上可能不如传统方法

## 4. 算法实现

本节详细介绍项目中实现的三种主要分类算法：朴素贝叶斯、支持向量机(SVM)和基于Word2Vec的LSTM深度学习模型。

### 4.1 朴素贝叶斯模型

#### 4.1.1 算法简介

朴素贝叶斯算法是一种基于贝叶斯定理与特征条件独立假设的概率分类器。在文本分类中，它通过计算文档属于各类别的概率来进行分类，其核心思想基于贝叶斯定理：

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

其中，$P(y|X)$ 是给定特征向量 $X$ 时，文档属于类别 $y$ 的后验概率；$P(X|y)$ 是似然概率，表示类别 $y$ 产生特征 $X$ 的概率；$P(y)$ 是类别 $y$ 的先验概率；$P(X)$ 是特征 $X$ 的边缘概率。

#### 4.1.2 算法思路

朴素贝叶斯之所以称为"朴素"，是因为它假设所有特征之间相互独立。在文本分类中，这意味着假设文档中的每个词的出现与其他词无关。基于这一假设，可以将联合概率 $P(X|y)$ 简化为所有单独特征条件概率的乘积：

$$P(X|y) = \prod_{i=1}^{n} P(x_i|y)$$

其中，$x_i$ 是特征向量 $X$ 的第 $i$ 个元素。

#### 4.1.3 算法流程说明

朴素贝叶斯分类的基本流程如下：

1. **训练阶段**：
   - 计算每个类别的先验概率 $P(y)$
   - 计算每个类别下各特征的条件概率 $P(x_i|y)$
   - 对于文本分类，通常计算每个词在每个类别文档中出现的概率

2. **预测阶段**：
   - 对于新文档，计算它属于各类别的后验概率
   - 选择具有最高后验概率的类别作为预测结果

#### 4.1.4 算法实现

在本项目中，我们使用scikit-learn库中的MultinomialNB实现多项式朴素贝叶斯分类器，该实现特别适合文本分类任务：

```python
def train_nb(self, X_train, y_train):
    print("训练朴素贝叶斯模型...")
    self.nb.fit(X_train, y_train)

def predict_nb(self, X_test):
    return self.nb.predict(X_test)
```

MultinomialNB是专为多项式分布的数据设计的，非常适合表示词频的文本特征。

#### 4.1.5 模型结构和参数选择

朴素贝叶斯模型的主要参数是平滑参数 alpha（拉普拉斯平滑），用于处理训练集中未出现的词：

```python
self.nb = MultinomialNB()  # 默认 alpha=1.0
```

较大的alpha值提供更强的平滑效果，有助于减少过拟合，而较小的值则保留更多训练数据的特性。在本项目中，我们使用默认值1.0，这是一个常用且有效的选择。

#### 4.1.6 算法性能

朴素贝叶斯在文本分类任务上表现稳健，特别是当与TF-IDF特征结合时：

- **计算效率**：训练和预测速度极快，适合大规模数据集和实时应用
- **准确性**：在使用TF-IDF特征时，本项目达到了86.4%的准确率
- **优势**：实现简单，计算高效，对小样本数据表现良好
- **局限性**：由于独立性假设，无法捕捉词语之间的关系和上下文信息

### 4.2 支持向量机 (SVM)

#### 4.2.1 算法简介

支持向量机(Support Vector Machine, SVM)是一种强大的监督学习算法，旨在找到将不同类别数据点分开的最佳超平面。在二分类问题中，SVM寻找一个决策边界，使得边界到最近数据点（支持向量）的距离（间隔）最大化。

#### 4.2.2 算法思路

SVM的核心思想是最大化分类间隔，这可以通过求解以下优化问题实现：

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 $$
$$\text{s.t. } y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \forall i$$

其中，$\mathbf{w}$ 是法向量，决定超平面的方向；$b$ 是偏置项；$\mathbf{x}_i$ 是特征向量；$y_i$ 是类别标签（±1）。

对于线性不可分的数据，SVM引入核技巧，将数据映射到高维空间，使其在新空间中线性可分。

#### 4.2.3 算法流程说明

SVM的基本工作流程如下：

1. **训练阶段**：
   - 将训练数据转换为适合SVM的格式
   - 选择合适的核函数（本项目使用线性核）
   - 通过二次规划求解器找到最优超平面
   - 识别支持向量（对决策边界有影响的数据点）

2. **预测阶段**：
   - 将新数据点与支持向量比较
   - 确定数据点相对于决策边界的位置
   - 返回对应的类别标签

#### 4.2.4 算法实现

在本项目中，我们使用scikit-learn的LinearSVC实现，它对大型线性分类问题进行了优化：

```python
def train_svm(self, X_train, y_train):
    print("训练SVM模型...")
    self.svm.fit(X_train, y_train)

def predict_svm(self, X_test):
    return self.svm.predict(X_test)
```

LinearSVC实现使用了liblinear库，比使用libsvm的标准SVC更快，特别适合文本分类等高维问题。

#### 4.2.5 模型结构和参数选择

我们选择了以下SVM参数：

```python
self.svm = LinearSVC(random_state=42, max_iter=2000)
```

- **random_state=42**: 设置随机种子，确保结果可重复
- **max_iter=2000**: 最大迭代次数，增加确保收敛的机会
- **默认C=1.0**: 正则化参数，平衡间隔最大化和训练误差

线性核函数是文本分类的最佳选择之一，因为文本数据通常在高维空间中已经是线性可分的。

#### 4.2.6 算法性能

SVM在本项目中表现出色，尤其是与TF-IDF特征结合时：

- **准确性**：在TF-IDF特征上达到了89.1%的准确率
- **稳健性**：对于高维稀疏数据（如文本特征向量）有很好的泛化能力
- **优势**：在高维空间中表现优异，对于边缘案例不敏感
- **局限性**：对大规模数据集的训练时间较长，参数调优复杂

### 4.3 Word2Vec与LSTM深度学习模型

#### 4.3.1 算法简介

我们的深度学习方法结合了Word2Vec词嵌入和长短期记忆网络(LSTM)。Word2Vec是一种词嵌入技术，将词语映射到低维连续向量空间；LSTM是一种特殊的循环神经网络，能有效处理序列数据并捕捉长距离依赖关系。

#### 4.3.2 算法思路

该方法分两个关键步骤：

1. **词嵌入**：使用Word2Vec学习词的向量表示，捕捉词的语义和语法关系
2. **序列建模**：使用LSTM网络处理词序列，学习文本的上下文信息和情感表达模式

通过这种组合，模型能够理解词语的语义以及它们在评论中的顺序和上下文关系。

#### 4.3.3 算法流程说明

整体流程包括以下步骤：

1. **Word2Vec训练**：
   - 将所有评论文本分词
   - 使用Word2Vec算法训练词向量模型
   - 建立词汇表和索引映射

2. **文本转换**：
   - 将每条评论转换为词索引序列
   - 对序列进行填充或截断处理，使其长度一致

3. **LSTM模型训练**：
   - 设置嵌入层，初始化为预训练的Word2Vec权重
   - 构建双向LSTM网络
   - 使用反向传播和梯度下降优化模型

4. **预测与评估**：
   - 使用训练好的模型对测试集进行预测
   - 计算准确率、精确率、召回率和F1值

#### 4.3.4 算法实现

我们的LSTM模型架构实现如下：

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128, output_dim=2, n_layers=2, dropout=0.5, padding_idx=0):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_indices):
        # 词嵌入
        embedded = self.embedding(text_indices)
        
        # LSTM处理
        _, (hidden, _) = self.lstm(embedded)
        
        # 拼接双向LSTM的隐藏状态
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        # 全连接层分类
        return self.fc(hidden)
```

训练过程中，我们将文本转换为索引序列，并使用预训练的Word2Vec嵌入：

```python
def load_pretrained_embeddings(self, embedding_matrix):
    """加载预训练的词向量权重"""
    self.model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
```

#### 4.3.5 模型结构和参数选择

我们的LSTM模型使用以下参数：

- **词嵌入维度**: 100（与Word2Vec训练维度一致）
- **隐藏层大小**: 128
- **LSTM层数**: 2
- **双向LSTM**: 使用双向，可以同时捕捉前后文信息
- **Dropout率**: 0.5，防止过拟合
- **批量大小**: 64
- **训练轮数**: 40
- **优化器**: Adam，学习率为0.001

这些参数是基于经验和实验选择的，平衡了模型复杂度、训练效率和性能。

#### 4.3.6 算法性能

优化后的LSTM模型在测试集上表现最佳：

- **准确率**: 90.5%，超过了所有其他模型组合
- **精确率**: 90.3%，表明模型在预测正面评论时更准确
- **召回率**: 90.8%，表明模型能有效识别大部分正面评论
- **F1值**: 90.5%，表明模型在精确率和召回率上取得了良好平衡

与传统方法相比，LSTM+Word2Vec的优势在于：

- **捕捉序列信息**：理解词语顺序和上下文关系
- **学习语义关联**：通过词嵌入理解相似词的关系
- **处理复杂表达**：能识别复杂的情感表达模式（如否定、讽刺等）

主要缺点是训练时间较长，对计算资源要求较高。
