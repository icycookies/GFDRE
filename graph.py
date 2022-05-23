import numpy as np
import torch

import logging

logger = logging.getLogger(__name__)

class Graph(object):
    def __init__(self, num_nodes, num_node_types, num_edge_types, node_type=None, edge_index=[], edge_type=[]):
        self.num_nodes = num_nodes
        self.num_edges = 0
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        if node_type is not None:
            self.node_type = node_type
        else:
            self.node_type = np.zeros(self.num_nodes)
        self.edge_index = edge_index
        self.edge_type = edge_type

    def modify_node_type(self, u, u_type):
        assert u_type < self.num_node_types
        self.node_type[u] = u_type

    def add_edge(self, u, v, e_type):
        assert e_type < self.num_edge_types
        self.num_edges += 1
        self.edge_index.append([u, v])
        self.edge_type.append(e_type)

    def add_undirected_edge(self, u, v, e_type):
        self.add_edge(u, v, e_type)
        self.add_edge(v, u, e_type)

    def remove_duplicate_edges(self):
        self.to_array()
        sorted_edge_type = []
        sorted_edge_index = []
        for i in range(self.num_edge_types):
            index = np.where(self.edge_type == i)[0]
            if len(index) == 0:
                continue
            edges = np.sort(self.edge_index[index])
            for j in range(edges.shape[0]):
                if j > 0 and edges[j][0] == edges[j - 1][0] and edges[j][1] == edges[j - 1][1]:
                    continue
                sorted_edge_type.append(i)
                sorted_edge_index.append(edges[j])
        self.edge_index = sorted_edge_index
        self.edge_type = sorted_edge_type
        self.to_array()

    def add_self_loops(self, e_type):
        for i in range(self.num_nodes):
            self.add_edge(i, i, e_type)

    def to_array(self):
        self.node_type = np.array(self.node_type)
        self.edge_index = np.array(self.edge_index)
        self.edge_type = np.array(self.edge_type)

    def to_tensor(self):
        self.to_array()
        self.node_type = torch.LongTensor(self.node_type)
        self.edge_index = torch.LongTensor(self.edge_index)
        self.edge_type = torch.LongTensor(self.edge_type)

    def to(self, device):
        self.node_type = self.node_type.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        return self

    def print_all_edges(self, u):
        for i, e in enumerate(self.edge_index):
            if e[0] == u:
                print("%d->%d, %d" % (e[0], e[1], self.edge_type[i]))

    def print_example_edges(self, e_type):
        cnt = 0
        print("Edge type: %d" % (e_type))
        all_edges = ""
        for i, e in enumerate(self.edge_index):
            if self.edge_type[i] == e_type:
                cnt += 1
                all_edges += "%d->%d " % (e[0], e[1])
                if cnt >= 50:
                    break
        print("Edges: %s" % all_edges)

    def num_edges_each_type(self):
        cnt = [0] * self.num_edge_types
        for i in self.edge_type:
            cnt[i] += 1
        return np.array(cnt)
        
    def __call__(self):
        return (self.node_type, self.edge_index, self.edge_type)

def convert_graph_to_adj(graph, max_len):
    attention = np.zeros((graph.num_edge_types, max_len, max_len))
    for i, edge in enumerate(graph.edge_index):
        attention[graph.edge_type[i]][edge[0]][edge[1]] = 1
    return attention

def merge_graph(G1, G2):
    G = Graph(G1.num_nodes, G1.num_node_types, G1.num_edge_types + G2.num_edge_types, node_type=G1.node_type, edge_index=G1.edge_index, edge_type=G1.edge_type)
    for i, edge in enumerate(G2.edge_index):
        G.add_edge(edge[0], edge[1], G2.edge_type[i] + G1.num_edge_types)
    return G

def create_batched_graph(graphs, c):
    graphs[0].to_array()
    node_type = graphs[0].node_type
    edge_index = graphs[0].edge_index
    edge_type = graphs[0].edge_type
    all_num_nodes = c
    for i in range(1, len(graphs)):
        graphs[i].to_array()
        node_type = np.concatenate((node_type, graphs[i].node_type), axis=0)
        edge_index = np.concatenate((edge_index, graphs[i].edge_index + all_num_nodes), axis=0)
        edge_type = np.concatenate((edge_type, graphs[i].edge_type), axis=0)
        all_num_nodes += c
    return Graph(
        num_nodes=all_num_nodes,
        num_node_types=graphs[0].num_node_types,
        num_edge_types=graphs[0].num_edge_types,
        node_type=torch.tensor(node_type, dtype=torch.long), 
        edge_index=torch.tensor(edge_index, dtype=torch.long), 
        edge_type=torch.tensor(edge_type, dtype=torch.long)
    )

def build_graph(num_ents, tok2sent, tok2ent, tok_type):
    if len(tok2sent) <= 512:
        l = len(tok2sent)
        ent_pos = [[] for i in range(num_ents)]
        sent_pos = []
        g = Graph(512, num_node_types=3, num_edge_types=5, edge_index=[], edge_type=[])
        for i in range(l):
            g.modify_node_type(i, tok_type[i])
            if tok2ent[i] != -1:
                ent_pos[tok2ent[i]].append(i)
            if tok_type[i] == 1:
                sent_pos.append(i)
        for i in sent_pos:
            for j in sent_pos:
                if i != j:
                    g.add_edge(i, j, 0)                         # Sent-Sent
        for i in range(num_ents):
            if i + 1 >= len(tok2sent):
                break
            for j in range(num_ents):
                if j + 1 >= len(tok2sent):
                    break
                if i == j:
                    for u in ent_pos[i]:
                        for v in ent_pos[j]:
                            g.add_edge(u, v, 1)                 # Co-ref
                else:
                    for u in ent_pos[i]:
                        for v in ent_pos[j]:
                            if tok2sent[u] == tok2sent[v]:
                                g.add_edge(u, v, 2)             # Co-occur
                    # if tok_type[i + 1] == 2 and tok_type[j + 1] == 2:
                    #    g.add_edge(i + 1, j + 1, 4)             # Ent-Ent
            # for u in ent_pos[i]:
            #   if u != i and tok_type[u] == 2 and tok_type[i + 1] == 2:
            #       g.add_edge(i + 1, u, 4)                     # Ent-Men
        if g.num_edges == 0:
            g.add_edge(0, 0, 0)
        return g
    else:
        return [
            build_graph(num_ents, tok2sent[:512], tok2ent[:512], tok_type[:512]),
            build_graph(num_ents, [-1] + tok2sent[512:], [-1] + tok2ent[512:], [0] + tok_type[512:]),
        ]
