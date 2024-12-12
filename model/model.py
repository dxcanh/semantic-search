import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointEmbedding(nn.Module):

    def __init__(self, vocab_size, size):
        super(JointEmbedding, self).__init__()

        self.size = size

        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding(vocab_size, size)

        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        pos_tensor = self.attention_position(self.size, input_tensor)

        segment_tensor = torch.zeros_like(input_tensor).to(device)
        segment_tensor[:, sentence_size // 2 + 1:] = 1

        output = self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
        return self.norm(output)

    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)

        pos = torch.arange(sentence_size, dtype=torch.long).to(device)
        d = torch.arange(dim, dtype=torch.long).to(device)
        d = (2 * d / dim)

        pos = pos.unsqueeze(1)
        pos = pos / (1e4 ** d)

        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        return pos.expand(batch_size, *pos.size())

    def numeric_position(self, dim, input_tensor):
        pos_tensor = torch.arange(dim, dtype=torch.long).to(device)
        return pos_tensor.expand_as(input_tensor)


class AttentionHead(nn.Module):

    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()

        self.dim_inp = dim_inp

        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)

        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale

        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = f.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)

        return context


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()

        self.heads = nn.ModuleList([
            AttentionHead(dim_inp, dim_out) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        s = [head(input_tensor, attention_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)


class Encoder(nn.Module):

    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)  # batch_size x sentence size x dim_inp
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(BaseModel):

    def __init__(self, config):
        super(BERT, self).__init__()
        raise NotImplementedError   
        

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        raise NotImplementedError  
    
    def load_weights_from_huggingface(self,model_name):
        # Load weights manually from Hugging Face checkpoint
        # This example assumes 'model_name' is the path to the checkpoint
        # You'll need to load individual weights and set them to corresponding layers

        # Example: Loading weights for the JointEmbedding layer
        # Replace 'self.embedding.token_emb.weight' and 'self.embedding.segment_emb.weight' 
        # with the weights from the Hugging Face checkpoint
        self.embedding.token_emb.weight = nn.Parameter(torch.tensor(...))  # Load token_emb weights
        self.embedding.segment_emb.weight = nn.Parameter(torch.tensor(...))  # Load segment_emb weights

        # Load other weights for Encoder, token_prediction_layer, classification_layer, etc.
    
