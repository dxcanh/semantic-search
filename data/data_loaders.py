import os
import numpy as np
import pandas as pd
import torch
import logging
import gzip
import pickle
import tarfile
import json
import random
import requests

from datasets import Dataset, load_dataset
from itertools import chain
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from sentence_transformers import util
from sentence_transformers import InputExample
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url: str, save_path: str):
    """
    Tải xuống tệp từ URL và lưu vào đường dẫn được chỉ định
    
    Args:
        url (str): URL của tệp cần tải
        save_path (str): Đường dẫn lưu tệp
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Tải xuống thành công: {url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi khi tải xuống: {e}")
        raise

class MSMarcoDataset(Dataset):
    """
    MS MARCO data loading using Hugging Face's Dataset
    """

    def __init__(self, file_path: str, tokenizer, max_seq_length: int = 512, mlm_probability: float = 0.15, 
                 pad_to_max_length: bool = False, line_by_line: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.pad_to_max_length = pad_to_max_length

        logger.info(f"Đang tải dữ liệu từ {file_path}")
        try:
            self.raw_data = pd.read_csv(file_path, sep='\t', header=None)
            self.dataset = HFDataset.from_pandas(pd.DataFrame(self.raw_data.iloc[:, 0], columns=["text"]))

            logger.info("Đang mã hóa tập dữ liệu")
            self.dataset = self.dataset.map(self.tokenize_function, remove_columns=["text"], batched=True)

            if not line_by_line:
                logger.info("Nhóm văn bản thành các khối")
                self.dataset = self.dataset.map(self.group_texts, batched=True, num_proc=1)
        
        except Exception as e:
            logger.error(f"Lỗi khi xử lý dữ liệu: {e}")
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=True,
        )

    def group_texts(self, examples):
        """
        Nối các văn bản và chia thành các khối có max_seq_length.
        """
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.max_seq_length) * self.max_seq_length
        result = {
            k: [t[i: i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

class MSMacroTriplet(Dataset):
    """
    Custom MSMARCO dataset trả về bộ ba (truy vấn, dương, âm)
    """

    def __init__(self, 
                 queries: Dict[int, Dict[str, Any]], 
                 corpus: Dict[int, str], 
                 ce_scores: Dict[Tuple[int, int], float],
                 num_negs_per_system: int = 1, 
                 negs_to_use: Optional[List[str]] = None, 
                 use_all_queries: bool = True,
                 max_queries: Optional[int] = None):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        
        # Giới hạn số lượng truy vấn nếu max_queries được chỉ định
        if max_queries is not None:
            self.queries_ids = self.queries_ids[:max_queries]
        
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.num_negs_per_system = num_negs_per_system
        self.negs_to_use = negs_to_use
        self.use_all_queries = use_all_queries

        logger.info("Xáo trộn các mẫu âm cho mỗi truy vấn")
        for qid in self.queries_ids:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, index: int) -> InputExample:
        query = self.queries[self.queries_ids[index]]
        query_text = query['query']
        qid = query['qid']

        # Chọn ví dụ dương
        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)
            pos_text = self.corpus.get(pos_id, "")
            query['pos'].append(pos_id)
            
            if not pos_text:
                logger.warning(f"Không tìm thấy văn đoạn dương có ID {pos_id}")
                return self.__getitem__((index + 1) % len(self))
        else:
            logger.warning(f"Không có ví dụ dương cho truy vấn ID {qid}")
            return self.__getitem__((index + 1) % len(self))

        # Chọn ví dụ âm
        if len(query['neg']) > 0:
            neg_id = query['neg'].pop(0)
            neg_text = self.corpus.get(neg_id, "")
            query['neg'].append(neg_id)
            
            if not neg_text:
                logger.warning(f"Không tìm thấy văn đoạn âm có ID {neg_id}")
                return self.__getitem__((index + 1) % len(self))
        else:
            logger.warning(f"Không có ví dụ âm cho truy vấn ID {qid}")
            return self.__getitem__((index + 1) % len(self))

        pos_score = self.ce_scores.get((qid, pos_id), 0.0)
        neg_score = self.ce_scores.get((qid, neg_id), 0.0)

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score - neg_score)

    def __len__(self) -> int:
        return len(self.queries_ids)

def prepare_msmarco_data(
    data_folder: str = 'msmarco-data', 
    max_passages: int = 1000, 
    num_negs_per_system: int = 2
):
    """
    Chuẩn bị dữ liệu MSMARCO
    """
    os.makedirs(data_folder, exist_ok=True)

    # Tải và giải nén tập dữ liệu
    def safe_download_and_extract(file_name, url, extract=True):
        file_path = os.path.join(data_folder, file_name)
        if not os.path.exists(file_path):
            logger.info(f"Tải xuống {file_name}")
            download_file(url, file_path)
            
            if extract and file_path.endswith('.tar.gz'):
                with tarfile.open(file_path, "r:gz") as tar:
                    tar.extractall(path=data_folder)

        return file_path

    # Tải collection
    collection_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz'
    collection_path = safe_download_and_extract('collection.tar.gz', collection_url)
    collection_filepath = os.path.join(data_folder, 'collection.tsv')

    # Tải queries
    queries_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz'
    safe_download_and_extract('queries.tar.gz', queries_url)
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')

    # Tải cross-encoder scores
    ce_scores_url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
    safe_download_and_extract('cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_url, extract=False)
    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')

    # Tải hard negatives
    hard_negatives_url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
    safe_download_and_extract('msmarco-hard-negatives.jsonl.gz', hard_negatives_url, extract=False)
    hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')

    # Đọc corpus
    logger.info("Đọc corpus: collection.tsv")
    corpus = {}
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            try:
                pid, passage = line.strip().split("\t", 1)
                pid = int(pid)
                corpus[pid] = passage
            except Exception as e:
                logger.warning(f"Lỗi khi đọc dòng corpus: {e}")

    # Đọc queries
    logger.info("Đọc queries: queries.train.tsv")
    queries = {}
    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            try:
                qid, query = line.strip().split("\t", 1)
                qid = int(qid)
                queries[qid] = query
            except Exception as e:
                logger.warning(f"Lỗi khi đọc dòng query: {e}")

    # Đọc cross-encoder scores
    logger.info("Tải CrossEncoder scores")
    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)

    # Xử lý hard negatives
    logger.info("Đọc hard negatives")
    train_queries = {}
    negs_to_use = None

    with gzip.open(hard_negatives_filepath, 'rt', encoding='utf8') as fIn:
        for line in tqdm(fIn, desc="Xử lý hard negatives"):
            if max_passages > 0 and len(train_queries) >= max_passages:
                break

            try:
                data = json.loads(line)
                pos_pids = data.get('pos', [])

                # Xử lý negative passages
                neg_pids = set()
                if negs_to_use is None:
                    negs_to_use = list(data['neg'].keys())
                    logger.info(f"Sử dụng negatives từ các hệ thống: {negs_to_use}")

                for system_name in negs_to_use:
                    if system_name not in data['neg']:
                        continue

                    system_negs = data['neg'][system_name]
                    negs_added = 0
                    for pid in system_negs:
                        if pid not in neg_pids:
                            neg_pids.add(pid)
                            negs_added += 1
                            if negs_added >= num_negs_per_system:
                                break

                if len(pos_pids) > 0 and len(neg_pids) > 0:
                    qid = data['qid']
                    train_queries[qid] = {
                        'qid': qid,
                        'query': queries.get(qid, ""),
                        'pos': pos_pids,
                        'neg': list(neg_pids)
                    }
            except Exception as e:
                logger.warning(f"Lỗi khi xử lý dòng hard negatives: {e}")

    logger.info(f"Tổng số truy vấn huấn luyện: {len(train_queries)}")

    return train_queries, corpus, ce_scores, negs_to_use

if __name__ == "__main__":
    # Chuẩn bị dữ liệu
    train_queries, corpus, ce_scores, negs_to_use = prepare_msmarco_data()

    # Sử dụng tokenizer MiniLM
    tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
    
    # Khởi tạo tập dữ liệu triplet
    train_biencoder_dataset = MSMacroTriplet(
        queries=train_queries, 
        corpus=corpus, 
        ce_scores=ce_scores,
        num_negs_per_system=2, 
        negs_to_use=negs_to_use,
        max_queries=10  # Giới hạn để kiểm tra
    )

    # In thông tin mẫu
    if len(train_biencoder_dataset) > 0:
        print("Ví dụ huấn luyện đầu tiên:")
        first_example = train_biencoder_dataset[0]
        print(f"Truy vấn: {first_example.texts[0]}")
        print(f"Văn đoạn dương: {first_example.texts[1]}")
        print(f"Văn đoạn âm: {first_example.texts[2]}")
        print(f"Nhãn (pos_score - neg_score): {first_example.label}")
    else:
        logging.warning("Tập dữ liệu triplet huấn luyện trống.")