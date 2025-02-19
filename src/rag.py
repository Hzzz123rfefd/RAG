import argparse
import csv
import faiss
import numpy as np
from src.model import ModelBert,ModelRerank
import torch
from src.rought_rank import Faiss,TF_IDF
from src.utils import *

class RAG:
    def __init__(
        self, 
        knowledge_path = None,
        knowledge_vector_dim = 768,
        bert_name_or_path = "bert-base-chinese", 
        rerank_name_or_path = "BAAI/bge-reranker-base",
        device = "cuda"
    ):
        self.knowledge_vector_dim = knowledge_vector_dim
        self.knowledge_path = knowledge_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.bert_name_or_path = bert_name_or_path
        self.rerank_name_or_path = rerank_name_or_path
        self.knowledge_list = []
        self.knowledge_vector_list = []
        self.timer = Timer()

        with open(self.knowledge_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  
            for row in reader:
                self.knowledge_list.append(row[0])  
                vector = np.array([float(x) for x in row[1].split(',')])
                self.knowledge_vector_list.append(vector)
        
        # used to text2vec
        self.bert = ModelBert(
            model_name_or_path = self.bert_name_or_path,
            device = self.device
        )
        
        # used to rought recall -> faiss
        self.faiss = Faiss(knowledge_list = self.knowledge_list,knowledge_vector_list = self.knowledge_vector_list,  knowledge_dim = self.knowledge_vector_dim)
        
        # used to rought recall -> TF-IDF
        self.tf = TF_IDF(knowledge_list = self.knowledge_list)
        
        # used to precise recall -> rerank
        self.rerank = ModelRerank(
            model_name_or_path = 'BAAI/bge-reranker-base',
            device = self.device
        )

    def quiry(self, query, k1, k2, show_log = True):
        query_vector = self.bert.text_to_vector(query).numpy()
        rough_recall = []

        result_faiss = self.faiss.recall(query_vector, k = k1)
        result_tf = self.tf.recall(query,  k = k1)

        
        result = self.multi_channel_recall([result_faiss, result_tf], [0.5,0.5])
        for index, item in enumerate(result):
            rough_recall.append(item["text"])
            if index == k1 - 1:
                break

        rerank_result = self.rerank.recall(query, rough_recall, k = k2)
        knowledge_segments = [each["text"] for each in rerank_result]
        response = self.llm_recall(query,knowledge_segments)

        return response

    def llm_recall(self, query, knowledge_segments):
        knowledge_str = ""
        for idx, knowledge in enumerate(knowledge_segments, start=1):
            knowledge_str += f"{idx}. 知识片段 {idx}: {knowledge}\n"
        prompt = f"问题: {query}\n\n以下是相关的知识片段，帮助回答问题：\n\n{knowledge_str}请根据以上知识片段，结合问题，给出一个详细且准确的答案。"
        
        # TODO 
        answer = prompt
        return answer

    def multi_channel_recall(self, results: List[List[dict]], weights : List = None):
        result_dict = {}
        for results, weight in zip(results, weights):
            temp_dict = {item['text']: item['score'] for item in results}
            
            for text, score in temp_dict.items():
                if text in result_dict:
                    result_dict[text] += weight * score
                else:
                    result_dict[text] = weight * score
        merged_results = [{"text": text, "score": score} for text, score in result_dict.items()]
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        return merged_results