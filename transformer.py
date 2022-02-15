import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F
import copy


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, padding_idx):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.embed = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx, **kwargs)

    def forward(self, x):
        return self.embed(x)


class NormalPositionalEmbedding(nn.Embedding):

    def __init__(self, embedding_dim: int, num_embeddings=1024):
        kwargs = {'device': 'cpu', 'dtype': torch.float32}
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, **kwargs)

    def forward(self, input_ids_shape: torch.Size):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class PositionalEmbedding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.model_dim = model_dim
        self.dropout = nn.Dropout(dropout)

        positional_emb = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        w = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000) / model_dim))

        positional_emb[:, 0::2] = torch.sin(position * w)
        positional_emb[:, 1::2] = torch.cos(position * w)

        positional_emb = positional_emb.unsqueeze(0)
        self.register_buffer('positional_emb', positional_emb)

    def forward(self, embedding):
        embedding = embedding * math.sqrt(self.model_dim)
        seq_len = embedding.size(1)

        positional_emb = Variable(self.positional_emb[:, :seq_len], requires_grad=False)
        embedding = embedding + positional_emb

        return self.dropout(embedding)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attention_weight = None

        self.query_project = nn.Linear(embed_dim, embed_dim, **kwargs)
        self.key_project = nn.Linear(embed_dim, embed_dim, **kwargs)
        self.value_project = nn.Linear(embed_dim, embed_dim, **kwargs)
        self.out_matrix = nn.Linear(embed_dim, embed_dim, **kwargs)

        self.dropout = nn.Dropout(dropout)

    def _self_attention(self, query, key, value, attention_mask=None, dropout=None):
        """
        q: batch_size x heads x seq_length x d_model
        k: batch_size x heads x seq_length x d_model
        v: batch_size x heads x seq_length x d_model
        attention_mask: batch_size x 1 x seq_length
        output: batch_size x head x seq_length x d_model
        """

        batch_size, num_of_heads, seq_length, dim_head = query.shape
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * (dim_head ** -0.5)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

        attention_scores = nn.functional.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        attention_output = torch.matmul(attention_scores, value)
        return attention_output, attention_scores

    def _shape(self, tensor: torch.Tensor, sequence_length: int, batch_size: int):
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                    query,
                    key,
                    value,
                    attention_mask=None):

        batch_size, tgt_length, _ = query.shape
        _, src_length, _ = key.shape

        q = self.query_project(query)
        k = self.key_project(key)
        v = self.value_project(value)

        # change shape to (batch_size, number_of_heads, sequence_length, dim_head)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_output, self.attention_weight = self._self_attention(q, k, v, attention_mask, self.dropout)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, tgt_length, self.embed_dim)
        attention_output = self.out_matrix(attention_output)

        return attention_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim=4096, dropout=0.1):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}
        self.linear_1 = nn.Linear(embed_dim, ff_dim, **kwargs)
        self.activation_dropout = nn.Dropout(0.0)
        self.linear_2 = nn.Linear(ff_dim, embed_dim, **kwargs)
        self.ff_dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, hidden_states):
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ff_dropout(hidden_states)
        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}

        self.attention = MultiHeadAttention(num_heads, embed_dim, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-5, **kwargs)

        self.linear_1 = nn.Linear(embed_dim, 4096, **kwargs)
        self.activation = F.gelu
        self.activation_dropout = nn.Dropout(0.0)

        self.linear_2 = nn.Linear(4096, embed_dim, **kwargs)
        self.ff_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(embed_dim, eps=1e-5, **kwargs)

    def forward(self, hidden_states, encoder_attention_mask):
        # attention block
        residual = hidden_states
        hidden_states = self.attention(hidden_states, hidden_states, hidden_states, encoder_attention_mask)
        hidden_states = self.attention_dropout(hidden_states)

        # residual + normalization block
        hidden_states = residual + hidden_states
        hidden_states = self.attention_norm(hidden_states)

        # feed forward block
        residual = hidden_states
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ff_dropout(hidden_states)

        # residual + norm block
        hidden_states = residual + hidden_states
        hidden_states = self.ff_norm(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_encoder_layers, num_heads, dropout, embed_tokens):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}
        self.num_encoder_layers = num_encoder_layers
        self.word_embedding = embed_tokens
        self.positional_embedding = NormalPositionalEmbedding(embed_dim, 1024)
        # self.positional_embedding = PositionalEmbedding(embed_dim, dropout=dropout)
        self.norm_embedding = nn.LayerNorm(embed_dim, **kwargs)
        self.layers = nn.ModuleList(
            [copy.deepcopy(EncoderLayer(embed_dim, num_heads, dropout)) for _ in range(self.num_encoder_layers)])

    def forward(self, encoder_inputs, encoder_attention_mask):
        input_shape = encoder_inputs.size()

        # # embedding layer
        # word_embed = self.word_embedding(encoder_inputs)
        # hidden_states = self.positional_embedding(word_embed)
        # hidden_states = self.norm_embedding(hidden_states)

        # embedding layer
        word_embed = self.word_embedding(encoder_inputs)
        pos_embed = self.positional_embedding(input_shape)
        hidden_states = word_embed + pos_embed
        hidden_states = self.norm_embedding(hidden_states)

        # encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_attention_mask=encoder_attention_mask)

        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}

        self.self_attention = MultiHeadAttention(num_heads, embed_dim, dropout=dropout)
        self.self_attention_dropout = nn.Dropout(dropout)
        self.self_attention_norm = nn.LayerNorm(embed_dim, eps=1e-5, **kwargs)

        self.cross_attention = MultiHeadAttention(num_heads, embed_dim, dropout=dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_norm = nn.LayerNorm(embed_dim, eps=1e-5, **kwargs)

        self.linear_1 = nn.Linear(embed_dim, 4096, **kwargs)
        self.activation = F.gelu
        self.activation_dropout = nn.Dropout(0.0)
        self.linear_2 = nn.Linear(4096, embed_dim, **kwargs)
        self.ff_dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(embed_dim, eps=1e-5, **kwargs)

    def forward(self, hidden_states, encoder_outputs, decoder_self_attention_mask, decoder_cross_attention_mask):
        # self attention block
        residual = hidden_states
        hidden_states = self.self_attention(hidden_states, hidden_states, hidden_states, decoder_self_attention_mask)
        hidden_states = self.self_attention_dropout(hidden_states)

        # residual + normalization block
        hidden_states = residual + hidden_states
        hidden_states = self.self_attention_norm(hidden_states)

        # cross attention block
        residual = hidden_states
        hidden_states = self.cross_attention(hidden_states, encoder_outputs, encoder_outputs,
                                             decoder_cross_attention_mask)
        hidden_states = self.cross_attention_dropout(hidden_states)

        # residual + normalization block
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attention_norm(hidden_states)

        # feed forward block
        residual = hidden_states
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ff_dropout(hidden_states)

        # residual + norm block
        hidden_states = residual + hidden_states
        hidden_states = self.ff_norm(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_decoder_layers, num_heads, dropout, embed_tokens):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}
        self.num_decoder_layers = num_decoder_layers
        self.word_embedding = embed_tokens
        self.positional_embedding = NormalPositionalEmbedding(embed_dim, 1024)
        # self.positional_embedding = PositionalEmbedding(embed_dim, dropout=dropout)
        self.norm_embedding = nn.LayerNorm(embed_dim, eps=1e-5, **kwargs)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DecoderLayer(embed_dim, num_heads, dropout)) for _ in range(self.num_decoder_layers)])

    def forward(self, decoder_input, encoder_hidden_states, decoder_self_attention_mask, decoder_cross_attention_mask):
        input_shape = decoder_input.size()

        # # embedding layer
        # word_embed = self.word_embedding(decoder_input)
        # hidden_states = self.positional_embedding(word_embed)
        # hidden_states = self.norm_embedding(hidden_states)

        # embedding layer
        word_embed = self.word_embedding(decoder_input)
        pos_embed = self.positional_embedding(input_shape)
        hidden_states = word_embed + pos_embed
        hidden_states = self.norm_embedding(hidden_states)

        # decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, decoder_self_attention_mask,
                                  decoder_cross_attention_mask)

        return hidden_states

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_of_encoder_decoder, num_heads, dropout):
        super().__init__()
        kwargs = {'device': 'cpu', 'dtype': torch.float32}
        self.padding_idx = 1
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.padding_idx, **kwargs)
        self.encoder = Encoder(embed_dim, num_of_encoder_decoder, num_heads, dropout, self.word_embedding)
        self.decoder = Decoder(embed_dim, num_of_encoder_decoder, num_heads, dropout, self.word_embedding)
        self.final_output = nn.Linear(embed_dim, vocab_size, **kwargs)

    def forward(self, encoder_inputs, decoder_inputs, encoder_attention_mask, decoder_self_attention_mask):

        encoder_hidden_states = self.encoder(encoder_inputs, encoder_attention_mask)
        final_hidden_states = self.decoder(decoder_inputs, encoder_hidden_states,  decoder_self_attention_mask, encoder_attention_mask)

        output = self.final_output(final_hidden_states)
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
