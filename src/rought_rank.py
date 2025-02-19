import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

class Faiss:
    def __init__(self, knowledge_list = None, knowledge_vector_list = None, knowledge_dim = 768):
        self.knowledge_list = knowledge_list
        self.knowledge_vector_list = knowledge_vector_list
        self.index = faiss.IndexFlatL2(knowledge_dim)
        self.index.add(np.array(self.knowledge_vector_list))

    def recall(self, query_vector, k = 5):
        distances, indices = self.index.search(np.array([query_vector]), k)
        scores = 1 / (1 + distances)
        max_score = np.max(scores)
        min_score = np.min(scores)
        scores = (scores - min_score) / (max_score - min_score)
        results = [{"text": self.knowledge_list[int(idx)], "score": float(score)} for idx, score in zip(indices[0], scores[0])]
        return results

class TF_IDF:
    def __init__(self, knowledge_list = None):
        self.knowledge_list = knowledge_list
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.knowledge_list)

    def recall(self, query, k = 5):
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_k_indices = cosine_similarities.argsort()[-k:][::-1]
        top_k_similarities = cosine_similarities[top_k_indices]
        min_score = np.min(top_k_similarities)
        max_score = np.max(top_k_similarities)
        if max_score != min_score:
            scores = (top_k_similarities - min_score) / (max_score - min_score)
        else:
            scores = np.ones_like(top_k_similarities) * min_score
        results = [{"text": self.knowledge_list[int(idx)], "score": float(score)} for idx, score in zip(top_k_indices, scores)]
        return results