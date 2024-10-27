# %%
import sys
import os

wd = os.path.abspath('/vol/storage/GhostFreePro') #os.path.abspath('/home/icb/samuele.firmani/GhostFreePro')
## Add the directory to the Python path
sys.path.append(wd)
import GNN.global_variables as global_variables
sys.path.append(os.path.abspath(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline'))
sys.path.append(os.path.abspath(global_variables.home_folder + 'GhostFreePro/GNN'))
import GNN.models as models
from data_preprocessing_pipeline.preprocessing_utils import aggregate_scores
import numpy as np
import tqdm
import torch
from torch_geometric.data import Data
import tqdm
import torch
from typing import Tuple

# %%



def compute_page_rank_for_gene(gene: str, gene_indices_dict: dict, data: Data) -> torch.Tensor:
    print("Computing personalized page rank for gene: ", gene)
    teleport_probs = torch.zeros(data.num_nodes)
    teleport_probs[gene_indices_dict[gene]] = 1.0
    teleport_probs = teleport_probs.to(data.edge_index.device)
    print("Device: ", data.edge_index.device)
    ranks = models.page_rank(data, teleport_probs=teleport_probs, damping_factor=0.85, max_iterations=10000, tol=1e-8)
    return ranks #/ ranks.max()

def compute_personalized_page_rank(node_names_list: list, data: Data, gene_indices_dict: dict) -> Tuple[torch.Tensor, list]:
    list_of_corresponding_ranks = []
    # set tensors to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Reset device to: ", device)
    data = data.to(device)
    for gene in tqdm.tqdm(node_names_list):
        rank = compute_page_rank_for_gene(gene, gene_indices_dict, data)
        list_of_corresponding_ranks.append(rank)

    list_of_corresponding_ranks = torch.stack(list_of_corresponding_ranks, dim=1)
    return list_of_corresponding_ranks

data = torch.load('/vol/storage/artivir_data/gosemsim_matrices/data_object_go_sem_sim_squared.pt')
gene_names = np.loadtxt("/vol/storage/artivir_data/gosemsim_matrices/gene_names_ppr_goterms_filtered.txt", dtype=str)
gene_indices_dict = {node_name: index for index, node_name in enumerate(gene_names)}

# devide gene_names in 100 chunks of approximately the same size
#gene_names_chunks = np.array_split(gene_names, 10)
# %%
# parse one of the chunks to compute the corresponding ranks
#import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument('--chunk_index', type=int, default=0)
# %%
#args = parser.parse_args()
#index = args.chunk_index
#print('Processed chunk: ' + str(index)) 
#list_of_corresponding_ranks = compute_personalized_page_rank(node_names_list = gene_names_chunks[index], data = data, gene_indices_dict=gene_indices_dict)
#torch.save(list_of_corresponding_ranks, '/vol/storage/artivir_data/gosemsim_matrices/denbi_ranks/ranks_personalized_page_rank_go_sem_sim_' + str(index) + '.pt')

# %%
list_of_corresponding_ranks = compute_personalized_page_rank(node_names_list = gene_names, data = data, gene_indices_dict=gene_indices_dict)
torch.save(list_of_corresponding_ranks, '/vol/storage/artivir_data/gosemsim_matrices/denbi_ranks/ranks_personalized_page_rank_go_sem_sim_squared' + 'total' + '.pt')


# %%
list_of_corresponding_ranks = torch.load('/vol/storage/artivir_data/gosemsim_matrices/denbi_ranks/ranks_personalized_page_rank_go_sem_sim_' + 'total' + '.pt')
# %%
# remove the last column of the matrix, which corresponds to a duplicate
#list_of_corresponding_ranks = list_of_corresponding_ranks[:, :-1]

#torch.cuda.is_available()
#gene_names = np.load(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/gene_names_0.0.npy', allow_pickle=True)
# load gene names from txt file
gene_names = np.loadtxt(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/gene_names_ppr_goterms_filtered.txt', dtype=str)
# set to lower case
gene_names = np.array([gene.lower() for gene in gene_names])
# # gene_names

# %%
# read host factors from publications
import pandas as pd
host_factors = pd.ExcelFile(global_variables.home_folder + "GhostFreePro/data_preprocessing_pipeline/artivir_data/host_factors_from_publications.xlsx")
host_factors_dfs = {sheet_name: host_factors.parse(sheet_name) for sheet_name in host_factors.sheet_names}
potential_host_factors_list = host_factors_dfs["host_factors"]["Gene name"].str.lower().unique()

gene_list_antiviral = np.array(['gnrhr', 'oprd1', 'chrm3', 'hrh4', 'htr3b', 'itgav', 'kcnh2',
       'thra', 'oprk1', 'ntrk2', 'slc6a1', 'kit', 'htr6', 'scnn1g',
       'chrm1', 'scn11a', 'htr1e', 'f2', 'htr3c', 'htr1f', 'kcnh7',
       'esr2', 'gsto2', 'scnn1b', 'scnn1a', 'htr4', 'grin3a', 'nr1i3',
       'f2', 'gsto1', 'abl1', 'scnn1d', 'shbg', 'htr3d', 'chrm5', 'kcnq3',
       'noxo1', 'gstp1', 'pomc', 'ppara', 'gabrr2', 'oprm1', 'dhfr',
       'chrm2', 'htr3e', 'chrm4', 'scn5a', 'gabrr1', 'mttp', 'ntrk1',
       'itgb3', 'kcnh6', 'gabrr3'])

gene_list_proviral = np.array(['mmp2', 'mpo', 'ltf', 'neu2', 'akr1d1', 'ces1', 'rara', 'orm1',
       'prkcq', 'prkcg', 'bche', 'cacna1c', 'srd5a1', 'cacna1d',
       'cyp11b1', 'prkci', 'slc18a2', 'ache', 'prkcb', 'prkce', 'lpl',
       'tyms', 'enthd1', 'cacnb2', 'nos2', 'pde4b', 'rarb', 'cacna1h',
       'snca', 'nanos2', 'cyp1a2', 'ifnar2', 'pde4d', 'pde4c', 'cacna2d1',
       'cacna1g', 'rxrg', 'ces1', 'cacna1s', 'hsd11b1', 'ngf', 'xiap',
       'egfl7', 'ada', 'alox5', 'rxrb', 'pla2g2e', 'neu1', 'anxa1',
       'prkcz', 'nr0b1', 'pde4a', 'dsc1', 'prkcd', 'lcp1', 'cnga1',
       'cacna1i', 'rarg', 'scn10a', 'top2b', 'tnf', 'prkaca', 'top2a',
       'rxra', 'pde10a', 'srd5a2', 'cysltr1', 'kcnn3'])

functionally_validated_host_factors = np.load(global_variables.home_folder + "GhostFreePro/data_preprocessing_pipeline/artivir_data/strong_functionally_validated_host_factors.npy")
functionally_validated_host_factors = np.array([gene.lower() for gene in functionally_validated_host_factors])

# %%
indices_of_potential_host_factors = [index for index, gene_name in enumerate(gene_names) if gene_name in potential_host_factors_list]
indices_of_potential_not_host_factors = [index for index, gene_name in enumerate(gene_names) if gene_name not in potential_host_factors_list]
indices_of_antiviral = [index for index, gene_name in enumerate(gene_names) if gene_name in gene_list_antiviral]
indices_of_proviral = [index for index, gene_name in enumerate(gene_names) if gene_name in gene_list_proviral]
indices_of_functionally_validated = [index for index, gene_name in enumerate(gene_names) if gene_name in functionally_validated_host_factors]

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %%
# subtract 0.15 to the diagonal of the matrix to avoid self loops
list_of_corresponding_ranks = list_of_corresponding_ranks.to('cpu') - torch.diag(torch.ones(list_of_corresponding_ranks.shape[0])) * 0.15

# %%
tsne = TSNE(n_components=2, perplexity=50)
tsne.fit(list_of_corresponding_ranks.to('cpu').numpy().T)
# %%
fig = plt.figure(figsize=(40,40))
plt.scatter(tsne.embedding_[indices_of_potential_not_host_factors, 0], tsne.embedding_[indices_of_potential_not_host_factors, 1], c = "blue", label = "unlabeled", alpha = 0.5, s = 10)
plt.scatter(tsne.embedding_[indices_of_potential_host_factors, 0], tsne.embedding_[indices_of_potential_host_factors, 1], c = "red", label = "potential host factors", alpha=0.5, s = 20)
plt.scatter(tsne.embedding_[indices_of_antiviral, 0], tsne.embedding_[indices_of_antiviral, 1], c = "black", label = "antiviral", alpha=0.99, s = 100)
plt.scatter(tsne.embedding_[indices_of_functionally_validated, 0], tsne.embedding_[indices_of_functionally_validated, 1], c = "green", label = "functionally validated", alpha=0.99, s = 60)
plt.xlabel("t-SNE 1", fontsize=30)
plt.ylabel("t-SNE 2", fontsize=30)
plt.legend(fontsize=30)
plt.title("t-SNE of Local Positional Encoding obtained via Personalized Page rank on GoSemSim derived graph squared", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# %%

from torch_geometric.nn import Node2Vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load('/vol/storage/artivir_data/gosemsim_matrices/data_object_go_sem_sim.pt')
data = data.to(device)

model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=7, context_size=4, walks_per_node=5, num_negative_samples=1, sparse=True).to(torch.device('cpu'))
loader = model.loader(batch_size=128, shuffle=True, num_workers=8)

optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.005)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(torch.device("cpu")), neg_rw.to(torch.device("cpu")))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 100):
    loss = train()
    print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss))

# %%
model.eval()
z = model()
tsne = TSNE(n_components=2, perplexity=50)
tsne.fit(z.detach().numpy())
# %%
fig = plt.figure(figsize=(40,40))
plt.scatter(tsne.embedding_[indices_of_potential_not_host_factors, 0], tsne.embedding_[indices_of_potential_not_host_factors, 1], c = "blue", label = "unlabeled", alpha = 0.5, s = 10)
plt.scatter(tsne.embedding_[indices_of_potential_host_factors, 0], tsne.embedding_[indices_of_potential_host_factors, 1], c = "red", label = "potential host factors", alpha=0.5, s = 20)
plt.scatter(tsne.embedding_[indices_of_antiviral, 0], tsne.embedding_[indices_of_antiviral, 1], c = "black", label = "antiviral", alpha=0.99, s = 100)
plt.scatter(tsne.embedding_[indices_of_functionally_validated, 0], tsne.embedding_[indices_of_functionally_validated, 1], c = "green", label = "functionally validated", alpha=0.99, s = 60)
plt.xlabel("t-SNE 1", fontsize=30)
plt.ylabel("t-SNE 2", fontsize=30)
plt.legend(fontsize=30)
plt.title("t-SNE of Local Positional Encoding obtained via Node2Vec on GoSemSim derived graph", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# %%
