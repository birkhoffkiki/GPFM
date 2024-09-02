import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

import sys

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
    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

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

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

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

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)
    
def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

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
    
class BaseCMN(nn.Module):
    def __init__(self, in_dim = 1024, d_model = 512, d_ff = 512, num_heads = 8, dropout = 0.1, region_size = 256, topk = 512):
        super(BaseCMN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.topk = topk
        self.in_dim = in_dim
        self.cmm_size = 2048
        self.cmm_dim = 512

        self.cmn = MultiThreadMemory(num_heads, d_model, topk=topk)
        
        self.region_size = region_size
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

        self.model = HATEncoder(self.encoder_layout)

        self.memory_matrix = nn.Parameter(torch.FloatTensor(self.cmm_size, self.cmm_dim))
        nn.init.normal_(self.memory_matrix, 0, 1 / self.cmm_dim)
        
        self.L = self.d_model
        self.D = 128
        self.K = self.topk
        self.attn_mil = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        self.attn_mem = MultiHeadedAttention(self.num_heads, self.d_model)
        
        self.use_bn = 0
        self.input_encoding_size = self.d_model
        self.att_feat_size = 2048
        self.drop_prob_lm = 0.5
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.input_encoding_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

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
            seq_mask = (seq.data > 0) #! Pad token?
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def forward(self, att_feats, seq = None, att_masks=None):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, att_masks)
        return out