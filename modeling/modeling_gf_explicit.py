# explicit graph neural adapters
# parallel GNN layers
import torch
import torch.nn as nn
from transformers.modeling_bert import BertEncoder
from transformers.activations import gelu
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)

class GLMFEEncoder(BertEncoder):

    def __init__(self, config, k=2, num_node_type=4, num_edge_type=5, dropout=0.2, graph_hidden_size=768, ie_size=3072, p_fc=0, ie_layer_num=2, sep_ie_layers=True):
        super().__init__(config)

        self.k = k
        # self.gnn_layers = nn.ModuleList([GATConvE(graph_hidden_size, num_node_type, num_edge_type, self.edge_encoder) for _ in range(k)])
        self.gnn_layers = nn.ModuleList([
            RGCNConv(in_channels=config.hidden_size, out_channels=graph_hidden_size, num_relations=num_edge_type)
        for _ in range(self.k)])
        self.activation = GELU()
        self.dropout_rate = dropout
        self.ie_layer_num = ie_layer_num

        self.sent_size = config.hidden_size
        self.graph_hidden_size = graph_hidden_size
        self.sep_ie_layers = sep_ie_layers
        if sep_ie_layers:
            self.ie_layers = nn.ModuleList(
                [MLP(self.graph_hidden_size, ie_size, self.sent_size, ie_layer_num, p_fc) for _ in range(k)]
            )
            """
            self.ie_layers = nn.ModuleList(
                [nn.Linear(self.graph_hidden_size, self.sent_size) for _ in range(k)]
            )
            """
        else:
            self.ie_layer = MLP(self.sent_size + graph_hidden_size, ie_size, self.sent_size, ie_layer_num, p_fc, layer_norm=True)

        self.num_hidden_layers = config.num_hidden_layers

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
        """
        hidden_states: [bs, seq_len, sent_size]
        attention_mask: [bs, 1, 1, seq_len]
        head_mask: list of shape [num_hidden_layers]

        graph: {
            node_type: [bs * seq_len]
            edge_index: [2, num_batched_edges]
            edge_type: [num_batched_edges]
        } 
        """
        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                # _X = torch.matmul(aggr_matrix, hidden_states)
                _X = hidden_states.view(-1, hidden_states.size(2)).contiguous()
                _X = self.gnn_layers[gnn_layer_index](_X, graph.edge_index.transpose(0, 1), graph.edge_type)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training = self.training)

                X = _X.view(bs, -1, _X.size(1)) # [bs, seq_length, node_size]
                
                lm_layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions=output_attentions)
                
                """
                # X = torch.cat((lm_layer_outputs[0], X), 2)
                if self.sep_ie_layers:
                    X = self.ie_layers[gnn_layer_index](X)
                else:
                    X = self.ie_layer(X)
                """
                # hidden_states = X
                # hidden_states = ent_mask * X + (1 - ent_mask) * lm_layer_outputs[0]
                hidden_states = 0.5 * X + 0.5 * lm_layer_outputs[0]
                
                layer_outputs = (hidden_states, lm_layer_outputs[1]) if output_attentions else hidden_states
            else:
                # LM
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions=output_attentions)
                hidden_states = layer_outputs[0]

            # print(layer_outputs)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)