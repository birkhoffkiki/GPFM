from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('..')

# from .att_model import pack_wrapper, AttModel
from mil_models.hat_modules.att_model import pack_wrapper, AttModel

#* Modules for Hierarchical Attention Transformer
def transform_tokens2regions(hidden_states, num_regions, region_size):

    # transform sequence into regions
    patch_hidden_states = torch.reshape(hidden_states, (hidden_states.size(0), num_regions, region_size, hidden_states.size(-1)))
    # squash regions into sequence into a single axis (samples * regions, region_size, hidden_size)
    hidden_states_reshape = patch_hidden_states.contiguous().view(hidden_states.size(0) * num_regions,
                                                                region_size, patch_hidden_states.size(-1))

    return hidden_states_reshape

def transform_masks2regions(mask, num_regions, region_size):
    
    # transform sequence mask into regions
    patch_mask = torch.reshape(mask, (mask.size(0), num_regions, region_size))
    # squash regions into sequence into a single axis (samples * regions, region_size)
    mask_reshape = patch_mask.contiguous().view(mask.size(0) * num_regions, 1, region_size)

    return mask_reshape

def transform_sentences2tokens(seg_hidden_states, num_sentences, max_sentence_length):
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(seg_hidden_states.size(0) // num_sentences, num_sentences,
                                                        max_sentence_length, seg_hidden_states.size(-1))
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(hidden_states.size(0), num_sentences * max_sentence_length,
                                                    hidden_states.size(-1))
    return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # mask = x.new_ones(x.shape[:2], dtype=torch.long)
        # mask = mask.unsqueeze(-2)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class HATLayer(nn.Module):
    def __init__(self, heads = 8, d_model = 512, d_ff = 512, region_size = 256, use_region_encoder = True, use_WSI_encoder = False, dropout = 0.1, max_patch = 100000, first_layer = False):
        super().__init__()
        self.region_size = region_size
        self.max_patch = max_patch
        self.max_region = int(np.ceil(self.max_patch / self.region_size))
        self.first_layer = first_layer # if is the first HAT layer, interpolate global token and add region level positional encodings
        
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        self.d_ff = d_ff
        
        self.global_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.region_position_embeddings = PositionalEncoding(self.d_model, self.dropout, max_len = 100000)
        
        self.use_region_encoder = use_region_encoder
        self.use_WSI_encoder = use_WSI_encoder
        
        self.attn = MultiHeadedAttention(self.heads, self.d_model, self.dropout)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        c = copy.deepcopy
        if self.use_region_encoder:
            self.region_encoder = TransformerLayer(self.d_model, c(self.attn), c(self.ff), self.dropout)
        if self.use_WSI_encoder:
            self.WSI_encoder = TransformerLayer(self.d_model, c(self.attn), c(self.ff), self.dropout)
            self.position_embeddings = nn.Embedding(self.max_region + 1, self.d_model, padding_idx = 0) #! Usage?
    
    def forward(self, x, mask, num_regions):
        
        assert self.use_region_encoder == True or self.use_WSI_encoder == True, "One of the encoders needs to be used"
        
        # num_regions = int(np.ceil(x.shape[1] / (self.region_size - 1)))
        
        if self.first_layer: # add global token
            x, mask = self.interpolate_global_token(x, mask)
            
        if self.use_region_encoder:
            region_inputs = transform_tokens2regions(x, num_regions, self.region_size)
            region_masks = transform_masks2regions(mask, num_regions, self.region_size)
            region_inputs = self.region_position_embeddings(region_inputs)
            
            outputs = self.region_encoder(region_inputs, region_masks)
        else:
            outputs = x
        
        # print("region level encoder output:", outputs.shape) #[num_regions, region_size, d_model]
            
        if self.use_WSI_encoder:
            
            assert self.use_region_encoder == True, "Region encoder needs to be used before WSI encoder"
            
            region_global_tokens = outputs[:, ::self.region_size].clone()
            # print("region_global_tokens shape:", region_global_tokens.shape)
            # print("mask shape:", mask.shape)
            region_attention_mask = mask[:, :, ::self.region_size].clone()
            # print("region_attention_mask shape:", region_attention_mask.shape)
            #* Original code for region level positional encodings (Absolute Positional Encoding)
            # region_positions = torch.arange(1, num_regions + 1).repeat(outputs.size(0), 1).to(outputs.device)
            # region_global_tokens += self.position_embeddings(region_positions)
            #*
            #* Relative Positional Encoding
            region_global_tokens = region_global_tokens.view(1, region_global_tokens.size(0), region_global_tokens.size(2))
            region_global_tokens = self.region_position_embeddings(region_global_tokens)
            #*
            # print(region_global_tokens.shape)
            WSI_outputs = self.WSI_encoder(region_global_tokens, region_attention_mask)
            WSI_outputs = WSI_outputs.view(WSI_outputs.size(1), 1, WSI_outputs.size(2))
            # replace region representation tokens
            outputs[:, ::self.region_size] = WSI_outputs
            # Map the outputs to the original shape
            outputs = outputs.view(x.size(0), num_regions * self.region_size, outputs.size(-1))
        else:
            outputs = outputs.view(x.size(0), num_regions * self.region_size, outputs.size(-1))
        
        outputs_mask = mask
        
        # print(outputs.shape) # -> [bs, num_regions * region_size, d_model]
        return outputs, outputs_mask
    
    def interpolate_global_token(self, hidden_states, mask):
        batch_size, seq_len, hidden_dim = hidden_states.size()
        num_regions = int(np.ceil(seq_len / (self.region_size - 1)))

        # Calculate the total size after division and padding
        total_size = num_regions * self.region_size

        # Calculate the padding size
        padding_size = total_size - seq_len - num_regions

        # Pad the sequence and mask if needed
        if padding_size > 0:
            hidden_padding = torch.zeros(batch_size, padding_size, hidden_dim, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, hidden_padding], dim=1)

            mask_padding = torch.zeros(batch_size, 1, padding_size, device=mask.device)
            mask = torch.cat([mask, mask_padding], dim=2)

        # Add the global token at the end of each region
        global_token = self.global_token.repeat(batch_size, num_regions, 1)
        
        hidden_states_with_global = torch.cat([hidden_states.view(batch_size, num_regions, self.region_size - 1, hidden_dim),
                                            global_token.unsqueeze(2)], dim=2)
        hidden_states_with_global = hidden_states_with_global.view(batch_size, num_regions * self.region_size, hidden_dim)

        # Update mask for global tokens (1 for global tokens)
        global_token_mask = torch.ones(batch_size, 1, num_regions, 1, device=mask.device)
        mask_with_global = torch.cat([mask.view(batch_size, 1, num_regions, self.region_size - 1), 
                                    global_token_mask], dim=3)
        mask_with_global = mask_with_global.view(batch_size, 1, num_regions * self.region_size)

        return hidden_states_with_global, mask_with_global
    
class HATEncoder(nn.Module):
    def __init__(self, encoder_layout = None):
        super().__init__()
        self.encoder_layout = encoder_layout
        self.layer = nn.ModuleList([HATLayer(heads = encoder_layout['num_heads'], d_model = encoder_layout['d_model'], d_ff = encoder_layout['d_ff'], 
                                             region_size = encoder_layout['region_size'], use_region_encoder = encoder_layout[str(idx)]['region_encoder'], 
                                             use_WSI_encoder = encoder_layout[str(idx)]['WSI_encoder'], dropout = encoder_layout['dropout'], 
                                             first_layer = encoder_layout[str(idx)]['first_layer']) for idx in range(int(encoder_layout['num_layers']))])
        self.norm = LayerNorm(encoder_layout['d_model'])
        self.pooler = HATPooler(encoder_layout, pooling = encoder_layout['pooling'])
        self.region_size = encoder_layout['region_size']
        
    def forward(self, x, mask):
        
        num_regions = int(np.ceil(x.shape[1] / (self.region_size - 1)))
        
        for idx, layer in enumerate(self.layer):
            x, mask = layer(x, mask, num_regions)
        x = self.norm(x)
        
        # Take out the global token of each region and send to the decoder
        x = x.view(num_regions, self.region_size, x.size(-1))
        if self.encoder_layout['pooling'] == 'None':
            output = x[:, ::self.region_size].view(1, num_regions, x.size(-1))
            return output
        else:
            output = self.pooler(x)
            output = output.unsqueeze(0)
            return output

class AttentivePooling(nn.Module):
    def __init__(self, encoder_layout):
        super().__init__()
        self.attn_dropout = encoder_layout['dropout']
        self.lin_proj = nn.Linear(encoder_layout['d_model'], encoder_layout['d_model'])
        self.v = nn.Linear(encoder_layout['d_model'], 1, bias=False)

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attention_weights_normalized = torch.softmax(attention_weights, -1)
        return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)

class HATPooler(nn.Module):
    def __init__(self, encoder_layout, pooling = 'max'):
        super().__init__()
        self.dense = nn.Linear(encoder_layout['d_model'], encoder_layout['d_model'])
        self.pooling = pooling
        if self.pooling == 'attentive':
            self.attentive_pooling = AttentivePooling(encoder_layout)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if self.pooling == 'attentive':
            pooled_output = self.attentive_pooling(hidden_states)
        else:
            pooled_output = torch.max(hidden_states, dim=1)[0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output
#*

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn, d_model, num_heads):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.L = self.d_model
        self.D = 128
        self.K = 256
        self.attn_mil = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.attn_mem = MultiHeadedAttention(self.num_heads, self.d_model)

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None):
        embeddings = self.tgt_embed(tgt)

        # Memory querying and responding for textual features
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0), memory_matrix.size(1))
        # A = self.attn_mil(embeddings)
        # A = A.squeeze()
        # A = torch.transpose(A, -1, -2)
        # A = F.softmax(A, dim=-1)
        # tmp = embeddings.squeeze()
        # M = torch.mm(A, tmp)
        # M = M.unsqueeze(0)
        # responses = self.cmn(M, dummy_memory_matrix, dummy_memory_matrix)
        # response_mask = responses.new_ones(responses.shape[:2], dtype=torch.long)
        # response_mask = response_mask.unsqueeze(-2)
        # embeddings = embeddings + self.attn_mem(embeddings, responses, responses, response_mask)
        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix) # Original
        embeddings = embeddings + responses # Original
        # Memory querying and responding for textual features 

        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        #* AveragePooling
        pooling = nn.AdaptiveAvgPool2d((x.shape[1] // 8, x.shape[2]))
        if x.shape[1] > 3000:
            x = pooling(x)
        #*
        
        #* MaxPooling
        # if x.shape[1] > 3000:
        #     desired_seq_len = x.shape[1] // 8
        #     kernel_size = x.shape[1] // desired_seq_len
        #     pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)
        #     x = x.permute(0, 2, 1)  # Change dimensions to (batch_size, hidden_dim, seq_len)
        #     x = pooling(x)
        #     x = x.permute(0, 2, 1)  # Change dimensions back to (batch_size, seq_len, hidden_dim)
        #*
        mask = x.new_ones(x.shape[:2], dtype=torch.long)
        mask = mask.unsqueeze(-2)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        src_mask = m.new_ones(m.shape[:2], dtype=torch.long)
        src_mask = src_mask.unsqueeze(-2)
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)
        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) #!

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=70000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BaseCMN(AttModel):

    def make_model(self, tgt_vocab, cmn, encoder_layout = None):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            HATEncoder(encoder_layout),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn, self.d_model, self.num_heads)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, dropout, topk, region_size, k):
        super(BaseCMN, self).__init__()
        self.num_layers = 3
        self.d_model = 512
        self.d_ff = 512
        self.num_heads = 8
        self.dropout = dropout
        self.topk = topk
        self.region_size = region_size
        self.cmm_size = 2048
        self.cmm_dim = 512
        self.k = k
        tgt_vocab = self.vocab_size + 1

        self.cmn = MultiThreadMemory(self.num_heads, self.d_model, topk=self.topk)
        
        self.encoder_layout = {
                'num_heads': self.num_heads,
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'region_size': self.region_size,
                'dropout': self.dropout,
                'pooling': 'attentive',
                'num_layers': 2,
                '0': {
                    'region_encoder': True,
                    'WSI_encoder': True,
                    'first_layer': True
                },
                '1': {
                    'region_encoder': True,
                    'WSI_encoder': False,
                    'first_layer': False
                },
            }

        self.model = self.make_model(tgt_vocab, self.cmn, self.encoder_layout)
        self.logit = nn.Linear(self.d_model, tgt_vocab)

        self.memory_matrix = nn.Parameter(torch.FloatTensor(self.cmm_size, self.cmm_dim))
        nn.init.normal_(self.memory_matrix, 0, 1 / self.cmm_dim)
        
        self.L = self.d_model
        self.D = 128
        self.K = self.k
        self.attn_mil = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        self.attn_mem = MultiHeadedAttention(self.num_heads, self.d_model)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):

        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        # Memory querying and responding for visual features
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0), self.memory_matrix.size(1))
        
        #* Attn_MIL based visual token selection
        # A = self.attn_mil(att_feats)
        # A = A.squeeze()
        # A = torch.transpose(A, -1, -2)
        # A = F.softmax(A, dim=-1)
        # tmp = att_feats.squeeze()
        # M = torch.mm(A, tmp)
        # M = M.unsqueeze(0) 
        #*
        #* Uniform Sampling
        indices = torch.linspace(0, att_feats.shape[1] - 1, steps=self.K).long()
        M = att_feats[:, indices, :]
        #*
        responses = self.cmn(M, dummy_memory_matrix, dummy_memory_matrix)
        response_mask = responses.new_ones(responses.shape[:2], dtype=torch.long)
        response_mask = response_mask.unsqueeze(-2)
        att_feats = att_feats + self.attn_mem(att_feats, responses, responses, response_mask)
        # Memory querying and responding for visual features

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    # def _forward(self, fc_feats, att_feats, seq = None, att_masks=None):
    def _forward(self, att_feats, seq = None, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        # out = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix)
        out = self.model.encode(att_feats, att_masks)
        # outputs = F.log_softmax(self.logit(out), dim=-1)
        # return outputs
        return out

    def _save_attns(self, start=False):
        if start:
            self.attention_weights = []
        self.attention_weights.append([layer.src_attn.attn.cpu().numpy() for layer in self.model.decoder.layers])

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past,
                                      memory_matrix=self.memory_matrix)

        if not self.training:
            self._save_attns(start=len(state) == 0)
        return out[:, -1], [ys.unsqueeze(0)] + past
    