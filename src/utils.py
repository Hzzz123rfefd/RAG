import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

def pad_sequence(item, max_length,pad_token_id):

    seq_length = max_length
    padded_sequence = torch.ones((item["input_ids"].shape[0], max_length), dtype=item["input_ids"].dtype) * pad_token_id
    attention_mask = torch.zeros((item["attention_mask"].shape[0], max_length), dtype=item["attention_mask"].dtype)
    
    padded_sequence[:, :seq_length] = item["input_ids"]
    attention_mask[:, :seq_length] = item["attention_mask"]
    
    item["input_ids"] = padded_sequence
    item["attention_mask"] = attention_mask
    return item

def recursive_collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: recursive_collate_fn([b[key] for b in batch]) for key in batch[0]}
    else:
        return default_collate(batch)
    
class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                

from typing import List, Tuple

def normalize_scores(results: List[List[Tuple[str, float]]]) -> List[List[Tuple[str, float]]]:
    """
    对每个结果列表的分数归一化到 [0, 1] 范围。
    """
    normalized_results = []
    for result in results:
        scores = [score for _, score in result]
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            normalized_results.append([(item, 1.0) for item, _ in result])
        else:
            normalized_results.append([
                (item, (score - min_score) / (max_score - min_score))
                for item, score in result
            ])
    return normalized_results

def combine_scores(
    results: List[List[Tuple[str, float]]], 
    group: List[str], 
    weights: List[float]
) -> List[Tuple[str, float]]:
    if len(results) != len(weights):
        raise ValueError("Results and weights must have the same length.")

    normalized_results = normalize_scores(results)

    score_maps = []
    for result in normalized_results:
        score_maps.append({item: score for item, score in result})

    final_scores = []
    for sentence in group:
        total_score = 0.0
        for score_map, weight in zip(score_maps, weights):
            total_score += score_map.get(sentence, 0.0) * weight
        final_scores.append((sentence, total_score))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores

import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started. Call start() first.")
        self.end_time = time.perf_counter()

    def get_elapsed_time(self):
        if self.end_time is None:
            raise ValueError("Timer has not been stopped. Call stop() first.")
        # elapsed_time = (self.end_time - self.start_time) * 1_000_000  # 转换为微秒
        elapsed_time = (self.end_time - self.start_time)
        return elapsed_time

    def reset(self):
        self.start_time = None
        self.end_time = None

    def elapsed(self, func):
        def wrapper(*args, **kwargs):
            self.start()
            result = func(*args, **kwargs)
            self.stop()
            print(f"Function {func.__name__} executed in {self.get_elapsed_time():.2f} 微秒")
            return result
        return wrapper
    
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

        
def calculate_metrics(predictions, ground_truth):
    binary_predictions = [1 if pred == 0 else 0 for pred in predictions]
    binary_ground_truth = [1 if true == 0 else 0 for true in ground_truth]
    
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    precision = precision_score(binary_ground_truth, binary_predictions)
    recall = recall_score(binary_ground_truth, binary_predictions)
    f1 = f1_score(binary_ground_truth, binary_predictions)
    
    log_message = (
        "==== 模型性能评估 ====\n"
        f"准确率 (Accuracy):  {accuracy:.4f}\n"
        f"精确率 (Precision): {precision:.4f}\n"
        f"召回率 (Recall):    {recall:.4f}\n"
        f"F1 得分 (F1-Score): {f1:.4f}\n"
        "=====================\n"
    )
    
    return log_message, accuracy, precision, recall, f1


def cosine_similarity_manual(vec1, vec2):
    # 计算点积
    dot_product = torch.dot(vec1, vec2)
    
    # 计算L2范数
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    
    # 计算余弦相似度
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity