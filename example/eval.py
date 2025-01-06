import argparse
import numpy as np
import sys
import os
sys.path.append(os.getcwd())


results = [
    {
        "query": "What is the capital of France?",
        "pos": ["Paris is the capital of France.", "France has many famous cities."],
        "neg": ["The Eiffel Tower is in Paris.", "France is in Europe."],
        "retrieved": ["Paris is the capital of France.", "The Eiffel Tower is in Paris.", "France is in Europe."]
    },
    {
        "query": "Who is the president of the USA?",
        "pos": ["Joe Biden is the president of the USA."],
        "neg": ["The USA is a country in North America.", "Barack Obama was the 44th president."],
        "retrieved": ["Barack Obama was the 44th president.", "Joe Biden is the president of the USA.","The USA is a country in North America."]
    }
]

def eval_rag(results, k):
    """
    计算 MRR@K, Recall@K, Precision@K, NDCG@K 指标
    :param results: 输入数据 [{"query": ..., "pos": [...], "neg": [...], "retrieved": [...]}]
    :param k: 召回数量
    :return: 评估结果
    """
    mrr, recall, precision, ndcg = 0.0, 0.0, 0.0, 0.0
    total_queries = len(results)

    for result in results:
        pos_set = set(result["pos"])  # 正样本集合
        retrieved = result["retrieved"]  # rag召回k个
        num_relevant = len(pos_set)  # 正样本数量

        # 计算 MRR@K
        reciprocal_rank = 0
        for i, doc in enumerate(retrieved[:k], start=1):
            if doc in pos_set:
                reciprocal_rank = 1 / i
                break
        mrr += reciprocal_rank

        # 计算 Recall@K
        relevant_in_top_k = sum(1 for doc in retrieved[:k] if doc in pos_set)
        recall += relevant_in_top_k / num_relevant if num_relevant > 0 else 0

        # 计算 Precision@K
        precision += relevant_in_top_k / k

        # 计算 NDCG@K
        dcg = 0
        idcg = sum(1 / np.log2(i + 1) for i in range(1, min(num_relevant, k) + 1))
        for i, doc in enumerate(retrieved[:k], start=1):
            if doc in pos_set:
                dcg += 1 / np.log2(i + 1)
        ndcg += dcg / idcg if idcg > 0 else 0

    # 平均
    mrr /= total_queries
    recall /= total_queries
    precision /= total_queries
    ndcg /= total_queries

    return {"MRR@K": mrr, "Recall@K": recall, "Precision@K": precision, "NDCG@K": ndcg}

def main(args):
    # 读取数据集
    
    
    # rag系统进行召回
    ## TODO
    
    # 开始评估
    metrics = eval_rag(results, args.k)
    print(f"Evaluation Metrics @K={args.k}: {metrics}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",type = int, default = 3)
    args = parser.parse_args()
    main(args)