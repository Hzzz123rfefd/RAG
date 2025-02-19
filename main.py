import argparse
from src.utils import *
from src.rag import *
# input = "你好"

def main(args):
    rag = RAG(
        knowledge_path = args.knowledge_path,
        knowledge_vector_dim = 768,
        bert_name_or_path = "bert-base-chinese", 
        rerank_name_or_path = args.rerank_model_path,
        device = args.device
    )

    rag.quiry("RAG系统的优势", k1 = args.k1 ,k2 = args.k2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_path",type=str,default = "database/corpus_vector.csv")
    parser.add_argument("--rerank_model_path",type=str,default = "BAAI/bge-reranker-base")
    parser.add_argument("--k1",type = int,default = 10)
    parser.add_argument("--k2",type = int,default = 5)
    parser.add_argument("--device",type=str,default = "cuda")
    args = parser.parse_args()
    main(args)
