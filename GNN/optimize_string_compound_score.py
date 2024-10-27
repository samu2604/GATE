# %%
import global_variables
import sys
import os
sys.path.append(os.path.abspath(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline'))
sys.path.append(os.path.abspath(global_variables.home_folder + 'GhostFreePro/GNN'))
from preprocessing_utils import aggregate_scores
import numpy as np
import pandas as pd
import tqdm
import torch
from torch_geometric.data import Data
from positional_encoding import compute_positional_encoding

# %%

if __name__ == '__main__':
    # %%
    path_to_ppi = global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/df_string_ppi_complete.zip'
    is_page_rank = False
    connection_threshold = 0.31
    is_prior = True
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    torch_data_ppi = compute_positional_encoding(path_to_ppi, [], is_page_rank = is_page_rank, connection_threshold = connection_threshold, \
                                         is_prior = is_prior, weights = weights, is_return_data_object = True) 
    
    
# %%
# select weight indices where df_ppi.weight greater than df_ppi.weight.max() / 2

# %%
# select weight indices where torch_data_ppi.weight greater than torch_data_ppi.weight.max() / 2

# torch_data_ppi.edge_weight[torch_data_ppi.edge_weight > torch_data_ppi.edge_weight.max() / 2]
# torch_data_ppi.edge_index[:, torch_data_ppi.edge_weight > torch_data_ppi.edge_weight.max() / 2]
# torch_data_ppi.node_names[torch_data_ppi.edge_index[:, torch_data_ppi.edge_weight > torch_data_ppi.edge_weight.max() / 2][0].unique()]
from torch_geometric.utils import to_networkx
# num_nodes = torch_data_ppi.edge_index[:, torch_data_ppi.edge_weight > torch_data_ppi.edge_weight.max() / 2][0].unique().shape[0]
# data2nx = Data(edge_index = torch_data_ppi.edge_index[:, torch_data_ppi.edge_weight > torch_data_ppi.edge_weight.max() / 2], num_nodes = num_nodes)
# %%
nx_data = to_networkx(torch_data_ppi, to_undirected=True)

# %%
import networkx as nx
print(nx_data.degree[0])
centrality = nx.degree_centrality(nx_data)
print(centrality[0])
print(nx.clustering(nx_data, 0))
#communities_generator = nx.community.girvan_newman(nx_data)
#top_level_communities = next(communities_generator)





# %%
functionally_validated_host_factors = np.load(global_variables.home_folder + "GhostFreePro/data_preprocessing_pipeline/artivir_data/strong_functionally_validated_host_factors.npy")
functionally_validated_host_factors = [name.lower() for name in functionally_validated_host_factors]
# read host factors from publications
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



# %%
# find where the host factors are in the ppi
potential_host_factors_list_indices = np.where(np.isin(torch_data_ppi.node_names, potential_host_factors_list))[0]
functionally_validated_host_factors_indices = np.where(np.isin(torch_data_ppi.node_names, functionally_validated_host_factors))[0]
gene_list_antiviral_indices = np.where(np.isin(torch_data_ppi.node_names, gene_list_antiviral))[0]
gene_list_proviral_indices = np.where(np.isin(torch_data_ppi.node_names, gene_list_proviral))[0]
# all other indices
other_indices = np.setdiff1d(np.arange(torch_data_ppi.num_nodes), np.concatenate((potential_host_factors_list_indices, functionally_validated_host_factors_indices, gene_list_antiviral_indices, gene_list_proviral_indices)))
# %%
potential_host_factor_degree = []
potential_host_factor_clustering = []
for index in potential_host_factors_list_indices:
    potential_host_factor_degree.append(nx_data.degree[index])
    potential_host_factor_clustering.append(nx.clustering(nx_data, index))
# %%
potential_host_factor_centrality = np.array(list(nx.degree_centrality(nx_data).values()))[potential_host_factors_list_indices]

# %%

gene_list_antiviral_degree = []
gene_list_antiviral_clustering = []
for index in gene_list_antiviral_indices:
    gene_list_antiviral_degree.append(nx_data.degree[index])
    gene_list_antiviral_clustering.append(nx.clustering(nx_data, index))

gene_list_antiviral_centrality = np.array(list(nx.degree_centrality(nx_data).values()))[gene_list_antiviral_indices] 
# %%

gene_list_proviral_degree = []
gene_list_proviral_clustering = []

for index in gene_list_proviral_indices:
    gene_list_proviral_degree.append(nx_data.degree[index])
    gene_list_proviral_clustering.append(nx.clustering(nx_data, index))
    # compute betweenness centrality

gene_list_proviral_centrality = np.array(list(nx.degree_centrality(nx_data).values()))[gene_list_proviral_indices]

# %%
nx.betweenness_centrality(nx_data)[index]
# %%
functionally_validated_host_factors_degree = []
functionally_validated_host_factors_clustering = []
for index in functionally_validated_host_factors_indices:
    functionally_validated_host_factors_degree.append(nx_data.degree[index])
    functionally_validated_host_factors_clustering.append(nx.clustering(nx_data, index))

functionally_validated_host_factors_centrality = np.array(list(nx.degree_centrality(nx_data).values()))[functionally_validated_host_factors_indices]

# %%
print("degree ", "potential_host_factor_degree ", np.mean(potential_host_factor_degree), 
      " gene_list_antiviral_degree ", np.mean(gene_list_antiviral_degree), 
      " gene_list_proviral_degree ", np.mean(gene_list_proviral_degree),
      " functionally_validated_host_factors_degree ", np.mean(functionally_validated_host_factors_degree))
print("degree std", " potential_host_factor_degree ", np.std(potential_host_factor_degree), 
      " gene_list_antiviral_degree ", np.std(gene_list_antiviral_degree), 
      " gene_list_proviral_degree ", np.std(gene_list_proviral_degree),
      " functionally_validated_host_factors_degree ", np.std(functionally_validated_host_factors_degree))
# %%

print("clustering ", " clustering ", np.mean(potential_host_factor_clustering),
      " gene_list_antiviral_clustering ", np.mean(gene_list_antiviral_clustering), 
      " gene_list_proviral_clustering ", np.mean(gene_list_proviral_clustering),
      " functionally_validated_host_factors_clustering ", np.mean(functionally_validated_host_factors_clustering))

print("clustering std ", " potential_host_factor_clustering ", np.std(potential_host_factor_clustering),
       " gene_list_antiviral_clustering ", np.std(gene_list_antiviral_clustering),
       " gene_list_proviral_clustering ", np.std(gene_list_proviral_clustering),
       " functionally_validated_host_factors_clustering ", np.std(functionally_validated_host_factors_clustering))

# %%
print("centrality ", " potential_host_factor_centrality ", np.mean(potential_host_factor_centrality),
        " gene_list_antiviral_centrality ", np.mean(gene_list_antiviral_centrality),
        " gene_list_proviral_centrality " ,np.mean(gene_list_proviral_centrality),
        " functionally_validated_host_factors_centrality ", np.mean(functionally_validated_host_factors_centrality))
print("centrality std ", " potential_host_factor_centrality ", np.std(potential_host_factor_centrality),
        " gene_list_antiviral_centrality ", np.std(gene_list_antiviral_centrality),
        " gene_list_proviral_centrality ", np.std(gene_list_proviral_centrality),
        " functionally_validated_host_factors_centrality ", np.std(functionally_validated_host_factors_centrality))


# %%

def avg_distance(G, group1, group2):
    """
    Calculate the average shortest path distance between nodes of group1 and group2.
    """
    total_distance = 0
    count = 0
    
    for node1 in group1:
        for node2 in group2:
            if node1 != node2: # Ensure we're not comparing a node to itself.
                total_distance += nx.shortest_path_length(G, node1, node2)
                count += 1

    return total_distance / count

# %%
# select random indices from potential_host_factors_list_indices
np.random.seed(111)
# %%
print("avg distance ", " potential_host_factors ", avg_distance(nx_data, np.random.choice(potential_host_factors_list_indices, len(gene_list_antiviral_indices), replace = False), np.random.choice(potential_host_factors_list_indices, len(gene_list_antiviral_indices), replace = False))) 

# %%
print("avg distance ", " potential_host_factors Vs gene_list_antiviral ", avg_distance(nx_data, np.random.choice(potential_host_factors_list_indices, len(gene_list_antiviral_indices), replace = False), gene_list_antiviral_indices))

# %%
print("avg distance ", " gene_list_antiviral ", avg_distance(nx_data, gene_list_antiviral_indices, gene_list_antiviral_indices)) 
# %%
print("avg distance ", " gene_list_proviral ", avg_distance(nx_data, np.random.choice(gene_list_proviral_indices, len(gene_list_antiviral_indices), replace = False), np.random.choice(gene_list_proviral_indices, len(gene_list_antiviral_indices), replace = False)))
# %%

print("avg distance ", " gene_list_proviral Vs gene_list_antiviral ", avg_distance(nx_data, np.random.choice(gene_list_proviral_indices, len(gene_list_antiviral_indices), replace = False), gene_list_antiviral_indices))

# %%
print("avg distance ", "other indices Vs gene_list_antiviral ", avg_distance(nx_data, np.random.choice(other_indices, len(gene_list_antiviral_indices), replace = False), gene_list_antiviral_indices))

# %%
print("avg distance ", "other indices Vs potential_host_factors ", avg_distance(nx_data, np.random.choice(other_indices, len(gene_list_antiviral_indices), replace = False), np.random.choice(other_indices, len(gene_list_antiviral_indices), replace = False)))
# %%
print("avg distance ", "other ",  avg_distance(nx_data, np.random.choice(other_indices, len(gene_list_antiviral_indices), replace=False), np.random.choice(other_indices, len(gene_list_antiviral_indices), replace = False)))
# %%

# check number of disconnected components in the graph
nx.number_connected_components(nx_data)

# %%
nx_data.is_directed()
# %%
betweenness_centrality = nx.betweenness_centrality(nx_data)


# %%
