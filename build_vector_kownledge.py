import argparse
import csv
import os

from src.model import ModelBert
import torch
from docx import Document
import numpy

def read_word_file(file_path):
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip() != '']
    return paragraphs

def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    knowledges = read_word_file(args.corpus_path)
    corpus_path_vectors = []

    bert = ModelBert(
        model_name_or_path = "bert-base-chinese",
        device = device
    )

    for knowledge in knowledges:
        corpus_path_vectors.append(bert.text_to_vector(knowledge).cpu().numpy())

    if args.clear_corpus_vector or not os.path.exists(args.corpus_vector_path):
        if os.path.exists(args.corpus_vector_path):
            os.remove(args.corpus_vector_path)
        with open(args.corpus_vector_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["sentence", "vector"]) 
            for sentence, vector in zip(knowledges, corpus_path_vectors):
                vector_str = ",".join(map(str, vector))
                writer.writerow([sentence, vector_str])
        
    else: 
        with open(args.corpus_vector_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for sentence, vector in zip(knowledges, corpus_path_vectors):
                vector_str = ",".join(map(str, vector))
                writer.writerow([sentence, vector_str])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path",type = str,default = "data/corpus.docx")
    parser.add_argument("--corpus_vector_path",type = str, default = "database/corpus_vector.csv")
    parser.add_argument("--clear_corpus_vector",type = int, default = 0)
    parser.add_argument("--device",type=str,default = "cuda")
    args = parser.parse_args()
    main(args)



