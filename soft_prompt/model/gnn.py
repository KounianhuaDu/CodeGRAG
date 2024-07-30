import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.nn import GlobalAttentionPooling


class CodeGNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, temperature=1):
        super(CodeGNN, self).__init__()

        self.K = nn.Linear(3 * in_feat, hidden_feat)
        self.V = nn.Linear(3 * in_feat, hidden_feat)
        self.Q = nn.Linear(2 * in_feat, hidden_feat)
        self.W = nn.Linear(2 * in_feat + hidden_feat, hidden_feat)

        self.K2 = nn.Linear(4 * in_feat, hidden_feat)
        self.V2 = nn.Linear(4 * in_feat, hidden_feat)
        self.Q2 = nn.Linear(3 * in_feat, hidden_feat)
        self.W2 = nn.Linear(3 * in_feat + hidden_feat, hidden_feat)

        self.layernorm = nn.LayerNorm(hidden_feat)
        self.readout = GlobalAttentionPooling(gate_nn=nn.Linear(hidden_feat, 1))

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temperature = temperature

    def inter_contrast(self, codes, graphs):
        # obtain mask
        mask = torch.eye(codes.shape[0], dtype=torch.bool).to(codes.device)
        neg_mask = ~mask
        mask = mask.float()
        neg_mask = neg_mask.float()

        logits = F.cosine_similarity(codes.unsqueeze(1), graphs.unsqueeze(0), dim=2)

        # compute loss
        exp_logits = torch.exp(logits / self.temperature)
        positives = (exp_logits * mask).sum()
        negatives = (exp_logits * neg_mask).sum()
        loss = -torch.log(positives / negatives)
        loss = torch.sum(loss) / (codes.shape[0])
        return loss

    def intra_contrast(self, z_i, z_j):
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (
            ~torch.eye(
                batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device
            )
        ).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

    def get_graph_embedding(self, g):
        edge_embds = g.edata["h"]
        srcs, dsts = g.edges()
        # first round
        g.ndata["Q"] = self.Q(torch.cat((g.ndata["kind"], g.ndata["type"]), dim=1))
        g.edata["K"] = self.K(
            torch.cat((g.ndata["kind"][srcs], g.ndata["type"][srcs], edge_embds), dim=1)
        )
        g.edata["V"] = self.V(
            torch.cat((g.ndata["kind"][srcs], g.ndata["type"][srcs], edge_embds), dim=1)
        )

        g.apply_edges(fn.v_mul_e("Q", "K", "alpha"))
        g.edata["alpha"] = edge_softmax(g, g.edata["alpha"])
        g.edata["V"] = g.edata["alpha"] * g.edata["V"]

        g.update_all(fn.copy_e("V", "h_n"), fn.sum("h_n", "h_n"))

        g.ndata["h"] = self.W(
            torch.cat((g.ndata["h_n"], g.ndata["kind"], g.ndata["type"]), 1)
        )
        g.ndata["h"] = self.layernorm(g.ndata["h"])

        # second round
        g.ndata["Q"] = self.Q2(
            torch.cat((g.ndata["kind"], g.ndata["type"], g.ndata["h"]), dim=1)
        )
        g.edata["K"] = self.K2(
            torch.cat(
                (
                    g.ndata["kind"][srcs],
                    g.ndata["type"][srcs],
                    edge_embds,
                    g.ndata["h"][srcs],
                ),
                dim=1,
            )
        )
        g.edata["V"] = self.V2(
            torch.cat(
                (
                    g.ndata["kind"][srcs],
                    g.ndata["type"][srcs],
                    edge_embds,
                    g.ndata["h"][srcs],
                ),
                dim=1,
            )
        )

        g.apply_edges(fn.v_mul_e("Q", "K", "alpha"))
        g.edata["alpha"] = edge_softmax(g, g.edata["alpha"])
        g.edata["V"] = g.edata["alpha"] * g.edata["V"]

        g.update_all(fn.copy_e("V", "h_n1"), fn.sum("h_n1", "h_n1"))

        g.ndata["h1"] = self.W2(
            torch.cat(
                (g.ndata["h_n1"], g.ndata["h"], g.ndata["kind"], g.ndata["type"]), 1
            )
        )
        g.ndata["h1"] = self.layernorm(g.ndata["h1"])

        graph_vec = self.readout(g, g.ndata["h1"])
        return graph_vec


"""device = 'cuda'
model = CodeGNN(256,256)
model.load_state_dict(torch.load('../output/1024/model_dict_13'))
model.eval()

def construct_faiss_index(embeddings):
    pca = faiss.PCAMatrix(embeddings.shape[-1], 32)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index, pca

def search_with_faiss(query, data_list, index, pca, k):
    query = query.to(device)
    query_embed = model.get_graph_embedding(query)
    query_embed = query_embed.cpu().detach().numpy() 

    query_embed = np.expand_dims(query_embed, axis=0)
    
    distances, indices = index.search(query_embed, k)

    prompt_str_list = [str(data_list[idx.item()]) for idx in indices[0][:k]]
    prompt = '\n'.join(prompt_str_list)
    return prompt"""
