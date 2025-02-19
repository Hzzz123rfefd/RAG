# RAG
Build a RAG system

# Introduction
## Recall
### Rough rank
#### Faiss
该算法用于计算向量a和向量库中所有向量的余弦相似性，进而获取粗排结果
假设语料库向量`$C∈R^{d×n}$`，其中`$d$`为语料数量，`$n$`为语料向量维度
```
# 创建一个线性索引
index = faiss.IndexFlatL2(n)
# 将数据添加到索引中
index.add(C)
```
当问题向量`$Q∈R^{n}$`进入时，我们需要从`$C$`中召回`$k$`个相似向量
```
D, I = index.search(Q, k)
```

#### TF-IDF
TF-IDF（Term Frequency - Inverse Document Frequency）是一种常用的文本特征提取方法，用于评估某个单词对一份文档集或语料库的重要性。它通过衡量一个词在文档中的出现频率以及它在整个文档集合中的普遍程度来赋予该词权重
假设语料库向量（TF-IDF Matrix）`$TF-IDF Matrix∈R^{d×n}$`，其中`$d$`为语料数量，`$n$`为语料向量维度（知识库唯一单词数量，每个单词的TF-IDF）
```python
# 创建 TF-IDF 向量化器
vectorizer = TfidfVectorizer()
# 将文档转化为 TF-IDF 向量
tfidf_matrix = vectorizer.fit_transform(documents)
```
当问题Q进入时，我们需要计算其`$TF-IDF$`向量`$TF-IDF∈R^{n}$`，我们需要从`$TF-IDF Matrix$`中召回`$k$`个相似向量
```python
# 将查询转化为 TF-IDF 向量
query_vector = vectorizer.transform([query])
# 计算查询与每个文档的余弦相似度
cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix[1:]).flatten()
# 获取最相关的文档（粗召回）
top_k_indices = cosine_similarities.argsort()[-k:][::-1]
D = [(self.knowledge[i], cosine_similarities[i]) for i in top_k_indices]
```

```math
\begin{aligned}
  \text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中总词数}} \\
  \text{IDF}(t) = \log\left(\frac{N}{1 + \text{DF}(t)}\right) \\
  \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
```
计算一个知识库的TF-IDF Matrix：<br>
比如现在有知识库<br>
documents = [
    "Python is a programming language used in data science.",
    "Machine learning is a key topic in data science.",
    "I love Python for programming.",
    "Data science involves statistics and computer science.",
    "Python programming is popular for data analysis."
]
通过上述公式可以计算出`$TF-IDF Matrix∈R^{d×n}$`，其中`$d$`为语料数量，`$n$`为向量维度（知识库唯一单词数量，每个单词的TF-IDF）
