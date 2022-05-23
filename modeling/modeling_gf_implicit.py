import logging

import math
import torch
import torch.nn as nn

from transformers.activations import gelu, gelu_new, swish
from torch_geometric.nn import RGCNConv
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}

class GLMFISelfAttention(nn.Module):
    def __init__(self, config, layer_id, entity_structure):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value_gcn = RGCNConv(config.hidden_size, self.all_head_size, num_relations=5)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # ==================SSAN==================
        self.entity_structure = entity_structure
        print(self.entity_structure)
        if entity_structure[0] != 'none':
            num_structural_dependencies = entity_structure[2]
            if entity_structure[0] == 'decomp' or entity_structure[0] == 'compose':
                self.bias_layer_k = nn.ParameterList(
                    [nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_attention_heads, self.attention_head_size))) for _ in range(num_structural_dependencies)])
                self.bias_layer_q = nn.ParameterList(
                    [nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_attention_heads, self.attention_head_size))) for _ in range(num_structural_dependencies)])
            if entity_structure[0] == 'biaffine' or entity_structure[0] == 'compose':
                self.bili = nn.ParameterList(
                    [nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.num_attention_heads, self.attention_head_size, self.attention_head_size)))
                     for _ in range(num_structural_dependencies)])
            self.abs_bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.num_attention_heads)) for _ in range(num_structural_dependencies)])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        layer_idx,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        graph=None,
        graph_adj=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_bias=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
            if layer_idx >= 12:
                # print(hidden_states.view(-1, hidden_states.shape[2]).shape, graph.edge_index.shape, graph.edge_type.shape)
                mixed_value_layer = mixed_value_layer + self.value_gcn(hidden_states.view(-1, hidden_states.shape[2]), graph.edge_index.transpose(0, 1), graph.edge_type).view(hidden_states.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer).float()
        key_layer = self.transpose_for_scores(mixed_key_layer).float()
        value_layer = self.transpose_for_scores(mixed_value_layer).float()

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # ==================SSAN==================
        # add attentive bias according to structure prior
        # query_layer / key_layer:         biaffine: [n_head, hidden_per_head, hidden_per_head]   decomp: [n_head, hidden_per_head]
        # att_bias[i]:               [bs, n_head, seq, seq]
        # if layer_idx >= 4: # you can set specific layers here with entity structure if you want to.
        all_bias = []
        if self.entity_structure[0] != 'none':
            for i in range(self.entity_structure[1]):
                if self.entity_structure[0] == 'decomp':
                    attention_bias_q = torch.einsum("bnid,nd->bni", query_layer, self.bias_layer_k[i]).unsqueeze(-1).repeat(1, 1, 1, query_layer.size(2))
                    attention_bias_k = torch.einsum("nd,bnjd->bnj", self.bias_layer_q[i], key_layer).unsqueeze(-2).repeat(1, 1, key_layer.size(2), 1)
                    bias = (attention_bias_q + attention_bias_k + self.abs_bias[i][None, :, None, None]) * graph_adj[i]
                    attention_scores += bias
                elif self.entity_structure[0] == 'biaffine':
                    attention_bias = torch.einsum("bnip,npq,bnjq->bnij", query_layer, self.bili[i], key_layer)
                    bias = (attention_bias + self.abs_bias[i][None, :, None, None]) * graph_adj[i]
                    attention_scores += bias
                elif self.entity_structure[0] == 'compose':
                    attention_bias_q = torch.einsum("bnid,nd->bni", query_layer, self.bias_layer_k[i]).unsqueeze(-1).repeat(1, 1, 1, query_layer.size(2))
                    attention_bias_k = torch.einsum("nd,bnjd->bnj", self.bias_layer_q[i], key_layer).unsqueeze(-2).repeat(1, 1, key_layer.size(2), 1)
                    attention_bias_bi = torch.einsum("bnip,npq,bnjq->bnij", query_layer, self.bili[i], key_layer)
                    bias = (attention_bias_q + attention_bias_k + attention_bias_bi + self.abs_bias[i][None, :, None, None]) * graph_adj[i]
                if output_bias:
                    all_bias.append(torch.sum(torch.abs(bias), axis=(1, 2, 3)) / torch.sum(graph_adj[i], axis=(1, 2, 3))) 

        if output_bias:
            all_bias = torch.stack(all_bias).transpose(0, 1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if output_bias:
            outputs = outputs + (all_bias,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GLMFIAttention(nn.Module):
    def __init__(self, config, layer_id, entity_structure):
        super().__init__()
        self.self = GLMFISelfAttention(config, layer_id, entity_structure)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        layer_idx,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        graph=None,
        graph_adj=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_bias=None,
    ):
        self_outputs = self.self(
            layer_idx, hidden_states, attention_mask, head_mask, graph, graph_adj, encoder_hidden_states, encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_bias=output_bias
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GLMFILayer(nn.Module):
    def __init__(self, config, layer_id, entity_structure):
        super().__init__()
        self.attention = GLMFIAttention(config, layer_id, entity_structure)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = GMLFIAttention(config, layer_id)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        layer_idx,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        graph=None,
        graph_adj=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_bias=None,
    ):
        self_attention_outputs = self.attention(layer_idx, hidden_states, attention_mask, head_mask, graph, graph_adj, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_bias=output_bias)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions, output_hidden_states, output_bias,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class GLMFIEncoder(nn.Module):
    def __init__(self, config, entity_structure):
        super().__init__()
        self.layer = nn.ModuleList([GLMFILayer(config, i, entity_structure) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        graph=None,
        graph_adj=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_bias=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        all_bias = ()
        # prepare for split/multi -head attention
        # [n_structure, bs, seq, seq] -> [bs, n_structure, n_head, seq, seq]
        graph_adj = graph_adj.transpose(0, 1)[:, :, None, :, :].float()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                i, hidden_states, attention_mask, head_mask[i], graph, graph_adj, encoder_hidden_states, encoder_attention_mask, output_attentions, output_hidden_states, output_bias
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if output_bias:
                all_bias = all_bias + (layer_outputs[2],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        if output_bias:
            outputs = outputs + (all_bias,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions), (all_bias)