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

from datasets import Dataset, load_dataset
from itertools import chain
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from sentence_transformers import util
from sentence_transformers import InputExample  # Thêm InputExample từ sentence_transformers

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MSMarcoDataset(Dataset):
    """
    MS MARCO data loading using Hugging Face's Dataset
    """

    def __init__(self, file_path: str,
                tokenizer,
                max_seq_length: int = 512,
                mlm_probability: float = 0.15, 
                pad_to_max_length: bool = False,
                line_by_line: bool = False):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.pad_to_max_length = pad_to_max_length

        logger.info(f"Loading data from {file_path}")
        self.raw_data = pd.read_csv(file_path, sep='\t', index_col=0)
        self.raw_data = self.raw_data.iloc[:, 0].tolist()
        self.dataset = Dataset.from_pandas(pd.DataFrame(self.raw_data, columns=["text"]))

        logger.info("Tokenizing the dataset")
        self.dataset = self.dataset.map(self.tokenize_function, remove_columns=["text"], batched=True)

        if not line_by_line:
            logger.info("Grouping texts into chunks")
            self.dataset = self.dataset.map(self.group_texts, batched=True, num_proc=1)

        logger.info("Applying torch_call to prepare tensors")
        self.dataset = self.dataset.map(self.torch_call, batched=True, batch_size=1000)

    def __len__(self) -> int:
        return len(self.dataset)  # Sửa từ `raise len(self.dataset)` thành `return len(self.dataset)`

    def __getitem__(self, index: int):
        return self.dataset[index]

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,  # Sửa 'only_second' thành 'True' nếu cần
            max_length=self.max_seq_length,
            return_special_tokens_mask=True,
        )
    
    def group_texts(self, examples):
        """
        Concatenates texts and splits them into chunks of max_seq_length.
        """
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Đảm bảo tổng độ dài là bội số của max_seq_length
        total_length = (total_length // self.max_seq_length) * self.max_seq_length
        result = {
            k: [t[i: i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def torch_call(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Prepares masked language modeling inputs and labels.
        """
        batch = self.tokenizer.pad(
            examples, 
            return_tensors="pt",
            padding="max_length" if self.pad_to_max_length else False,
            pad_to_multiple_of=8 if not self.pad_to_max_length else None
        )

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares masked tokens for MLM.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Chỉ tính loss trên các token được mask

        # 80% thời gian thay thế token bằng [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% thời gian thay thế token bằng từ ngẫu nhiên
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 10% thời gian giữ nguyên token đã mask
        return inputs, labels

class MSMacroTriplet(Dataset):
    """
    Custom MSMARCO dataset that returns triplets (query, positive, negative)
    """

    def __init__(self, queries: Dict[int, Dict[str, Any]], corpus: Dict[int, str], ce_scores: Dict[Tuple[int, int], float],
                num_negs_per_system: int = 1, negs_to_use: Optional[List[str]] = None, use_all_queries: bool = True):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.num_negs_per_system = num_negs_per_system
        self.negs_to_use = negs_to_use
        self.use_all_queries = use_all_queries

        logger.info("Shuffling negatives for each query")
        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, index: int) -> InputExample:
        query = self.queries[self.queries_ids[index]]
        query_text = query['query']
        qid = query['qid']

        # Chọn ví dụ dương
        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)    # Lấy và thêm lại vào cuối danh sách
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:
            raise ValueError(f"No positive examples available for query ID {qid}")

        # Chọn ví dụ âm
        if len(query['neg']) > 0:
            neg_id = query['neg'].pop(0)    # Lấy và thêm lại vào cuối danh sách
            neg_text = self.corpus[neg_id]
            query['neg'].append(neg_id)
        else:
            raise ValueError(f"No negative examples available for query ID {qid}")

        pos_score = self.ce_scores.get((qid, pos_id), 0.0)
        neg_score = self.ce_scores.get((qid, neg_id), 0.0)

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score - neg_score)

    def __len__(self) -> int:
        return len(self.queries)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased")
    
    data_folder = 'msmarco-data'

    #### Read the corpus files, that contain all the passages. Store them in the corpus dict
    corpus = {}         # dict in the format: passage_id -> passage. Stores all existent passages
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            # Thay thế `util.http_get` bằng hàm tải xuống đơn giản
            url = 'https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz'
            response = util.http_get(url)  # Nếu `util.http_get` không tồn tại, hãy sử dụng `requests` hoặc phương pháp khác

            # Lưu tệp tải xuống
            with open(tar_filepath, 'wb') as f:
                f.write(response)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    logging.info("Read corpus: collection.tsv")
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t", 1)  # Thêm giới hạn chia để tránh lỗi nếu passage chứa tab
            pid = int(pid)
            corpus[pid] = passage


    ### Read the train queries, store in queries dict
    queries = {}        #dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            url = 'https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz'
            response = util.http_get(url)  # Thay thế `util.http_get` bằng phương pháp tải xuống khác nếu cần

            # Lưu tệp tải xuống
            with open(tar_filepath, 'wb') as f:
                f.write(response)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)


    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t", 1)  # Thêm giới hạn chia để tránh lỗi nếu query chứa tab
            qid = int(qid)
            queries[qid] = query


    # Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
    # to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
    ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
    if not os.path.exists(ce_scores_file):
        logging.info("Download cross-encoder scores file")
        url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz'
        response = util.http_get(url)  # Thay thế `util.http_get` bằng phương pháp tải xuống khác nếu cần

        # Lưu tệp tải xuống
        with open(ce_scores_file, 'wb') as f:
            f.write(response)

    logging.info("Load CrossEncoder scores dict")
    with gzip.open(ce_scores_file, 'rb') as fIn:
        ce_scores = pickle.load(fIn)

    # As training data we use hard-negatives that have been mined using various systems
    hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
    if not os.path.exists(hard_negatives_filepath):
        logging.info("Download hard negatives file")
        url = 'https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz'
        response = util.http_get(url)  # Thay thế `util.http_get` bằng phương pháp tải xuống khác nếu cần

        # Lưu tệp tải xuống
        with open(hard_negatives_filepath, 'wb') as f:
            f.write(response)

    logging.info("Read hard negatives train file")
    train_queries = {}
    negs_to_use = None
    max_passages = 1000  # Đặt giá trị mặc định hoặc tùy chỉnh theo nhu cầu
    num_negs_per_system = 2  # Đặt giá trị mặc định hoặc tùy chỉnh theo nhu cầu

    with gzip.open(hard_negatives_filepath, 'rt', encoding='utf8') as fIn:
        for line in tqdm(fIn, desc="Processing hard negatives"):
            if max_passages > 0 and len(train_queries) >= max_passages:
                break
            data = json.loads(line)

            # Get the positive passage ids
            pos_pids = data.get('pos', [])

            # Get the hard negatives
            neg_pids = set()
            if negs_to_use is None:
                # Nếu bạn không sử dụng args, hãy xác định negs_to_use ở đây
                negs_to_use = list(data['neg'].keys())
                logging.info(f"Using negatives from the following systems: {negs_to_use}")

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

    logging.info(f"Train queries: {len(train_queries)}")
    
    # Initialize the triplet dataset
    train_biencoder_dataset = MSMacroTriplet(queries=train_queries, corpus=corpus, ce_scores=ce_scores,
                                            num_negs_per_system=num_negs_per_system, negs_to_use=negs_to_use,
                                            use_all_queries=True)
    if len(train_biencoder_dataset) > 0:
        print("First training example:")
        first_example = train_biencoder_dataset[0]
        print(f"Query: {first_example.texts[0]}")
        print(f"Positive Passage: {first_example.texts[1]}")
        print(f"Negative Passage: {first_example.texts[2]}")
        print(f"Label (pos_score - neg_score): {first_example.label}")
    else:
        logging.warning("Train triplet dataset is empty.")