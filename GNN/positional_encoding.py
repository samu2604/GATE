# %%

%cd ..
%pwd

# %%

from global_variables import global_vars as global_variables
import sys
import os
sys.path.append(os.getcwd() + 'data_preprocessing_pipeline')
sys.path.append(os.getcwd() + 'GNN')
import models
from data_preprocessing_pipeline.preprocessing_utils import aggregate_scores
import numpy as np
import pandas as pd
import tqdm
import torch
from torch_geometric.data import Data
import argparse

# %%

def compute_positional_encoding(path_to_ppi: str, gene_names_ranges: list, is_page_rank: bool = True, connection_threshold: float = 0.0, is_prior: bool = True, weights: list = None, is_return_data_object: bool = False):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Device is " + str(device))

    scores = ['neighborhood', 'neighborhood_transferred', 
    'fusion', 'cooccurence', 'coexpression', 
    'coexpression_transferred', 'experiments',
    'experiments_transferred', 'database', 
    'database_transferred', 'textmining',
    'textmining_transferred']

    if weights is None:
        weights = np.ones(len(scores))
    else:
        weights = np.array(weights)

    lists_of_scores_to_combine = [scores]#[['neighborhood', 'neighborhood_transferred'],
                                 # ['fusion'],
                                 # ['cooccurence'],
                                 # ['coexpression', 'coexpression_transferred'],
                                 # ['experiments', 'experiments_transferred'],
                                 # ['database', 'database_transferred'],
                                 # ['textmining', 'textmining_transferred'],
                                 # scores]
    list_of_corresponding_ranks = []
    randomized_gene_list = []
    for scores_to_combine in lists_of_scores_to_combine:
        scores_weights_dict = {}
        weights = np.ones(len(scores))
        print("Combining scores: " + str(scores_to_combine))
        for score, weight in zip(scores, weights):
            if score in scores_to_combine:
                scores_weights_dict[score] = weight
            else:
                scores_weights_dict[score] = 0
        # check if the data file is already computed and saved t file
        if os.path.isfile(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/data_object_' + str(connection_threshold) + "_is_prior_" + str(is_prior) + ".pt"):
            print("Data object already computed, loading it from file")
            data = torch.load(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/data_object_' + str(connection_threshold) + "_is_prior_" + str(is_prior) + ".pt")
            randomized_gene_list = data.node_names
        else:
            df_ppi = aggregate_scores(path_to_ppi, weights=list(scores_weights_dict.values()))
            edge_index_0 = []
            edge_index_1 = []
            edge_weights = []
            df_ppi = df_ppi[['gene_name_1', 'gene_name_2', 'combined_score_recomputed', 'combined_score']]
            df_ppi = df_ppi[df_ppi["combined_score"] > int(connection_threshold*1000)]
            df_ppi.reset_index(drop = True, inplace = True)
            print("The number of connections is " + str(len(df_ppi)))
            randomized_gene_list = df_ppi["gene_name_1"].unique() #this operation preserve the original order, it can also be randomized with np.random.permutation

            # create a lookup dictionary to get indices from gene names
            gene_indices_dict = {gene_name: index for index, gene_name in enumerate(randomized_gene_list)}
            gene_name_1_array = df_ppi["gene_name_1"].values
            gene_name_2_array = df_ppi["gene_name_2"].values
            combined_score_array = df_ppi["combined_score_recomputed"].values
            del df_ppi
            for gene_1, gene_2, score in tqdm.tqdm(zip(gene_name_1_array, gene_name_2_array, combined_score_array)):
                edge_index_0.append(gene_indices_dict[gene_1])
                edge_index_1.append(gene_indices_dict[gene_2])
                edge_weights.append(score)
            # set all weights in edge_weights that are smaller or equal to 0.041 to 0
            edge_weights = np.array(edge_weights)
            if not is_prior:
                edge_weights[edge_weights <= 0.041] = 1e-6
            edge_weights = edge_weights.tolist()

            network = [edge_index_0, edge_index_1, edge_weights]

            data = Data()
            data.edge_index = torch.Tensor([network[0], network[1]]).long()
            data.edge_weight = torch.Tensor(network[2])
            data.num_nodes = len(randomized_gene_list)
            data.node_names = randomized_gene_list

            if is_return_data_object:
                return data

            #save data object to file
            torch.save(data, global_variables.storage_folder + '/ppr/data_object_' + str(connection_threshold) + "_is_prior_" + str(is_prior) + ".pt")

        if is_page_rank:
            if gene_names_ranges[1] > len(randomized_gene_list):
                gene_names_ranges[1] = len(randomized_gene_list)
            # create a lookup dictionary to get indices from gene names
            gene_indices_dict = {gene_name: index for index, gene_name in enumerate(randomized_gene_list)}
            for gene in tqdm.tqdm(randomized_gene_list[gene_names_ranges[0]:gene_names_ranges[1]]):
                print("Computing Personalized PageRank for gene " + gene)
                teleport_probs = torch.zeros(data.num_nodes)
                teleport_probs[gene_indices_dict[gene]] = 1.0 # setting teleport probability to 1 for the gene of interest
                ranks = models.page_rank(data, teleport_probs = teleport_probs, damping_factor=0.85, max_iterations=10000, tol=1e-8)
                ranks = ranks / ranks.max() # normalize ranks to be between 0 and 1 TODO check if this is necessary
                #print(f"Ranks: {ranks.size()} and {ranks}")
                #print("std deviation:", np.std(ranks.numpy()))
                list_of_corresponding_ranks.append(ranks)

        else:
            # Train Node2Vec model to get embeddings of nodes in the graph (i.e. proteins in the PPI network) 
            model = models.Node2Vec(data=data, embedding_dim=32, walk_length=5,
                                    context_size=3, walks_per_node=5, p=1, q=1,
                                    num_negative_samples=1, sparse=True, 
                                    num_nodes=data.num_nodes)
            model = model.to(device)

            loader = model.loader(batch_size=10, shuffle=True, num_workers=4)
            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
            def train():
                model.train()
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                return total_loss / len(loader)

            for epoch in range(40):
                loss = train()
                print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
            # Get embeddings
            list_of_corresponding_ranks.append(model().detach().cpu())

    return list_of_corresponding_ranks, randomized_gene_list[gene_names_ranges[0]:gene_names_ranges[1]]


if __name__ == "__main__":
    path_to_ppi = global_variables.storage_folder + "9606.protein.links.full.v12.0.txt.gz"
    is_page_rank = True
    gene_names_ranges = [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000]
    parser = argparse.ArgumentParser(description='Compute positional encoding.')
    parser.add_argument('--start_index', type=int, help='Start index for gene_names_ranges')
    parser.add_argument('--end_index', type=int, help='End index for gene_names_ranges')
    args = parser.parse_args()

    i, j = args.start_index, args.end_index
    connection_threshold = 0.3
    is_prior = True
    list_of_corresponding_ranks, gene_names = compute_positional_encoding(path_to_ppi, gene_names_ranges[i:j+1], is_page_rank = is_page_rank, connection_threshold = connection_threshold, is_prior = is_prior)
    ranks = torch.stack(list_of_corresponding_ranks, dim = 1)
    print(ranks.shape)
    if is_page_rank:
        #torch.save(list_of_corresponding_ranks, global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/ranks_page_rank_' + str(connection_threshold) + "_is_prior_" + str(is_prior) + ".pt")
        print("Saving ranks to file " + global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_' + str(connection_threshold) + "_is_prior_" + str(is_prior) + "_indices_" + str(gene_names_ranges[i:j+1]) + "v12.pt")
        torch.save(list_of_corresponding_ranks, global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_' + str(connection_threshold) + "_is_prior_" + str(is_prior) + "_indices_" + str(gene_names_ranges[i:j+1]) + "v12.pt") 
    else:
        print("Saving ranks to file " + global_variables.storage_folder + 'ppr/ranks_node2vec_is_prior_' + str(is_prior) + ".pt")
        torch.save(list_of_corresponding_ranks, global_variables.storage_folder + 'ppr/ranks_node2vec_is_prior_' + str(is_prior) + '.pt')
    print("Saving gene names to file " + global_variables.storage_folder + 'ppr/gene_names_' + str(connection_threshold) + '_indices_' + str(gene_names_ranges[i:j+1]) + 'v12.npy')
    np.save(global_variables.storage_folder + 'ppr/gene_names_' + str(connection_threshold) + '_indices_' + str(gene_names_ranges[i:j+1]) + 'v12.npy', gene_names)

# %%
ranks01 = torch.load(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_is_prior_True_indices_[0, 3000]v12.pt')
ranks12 = torch.load(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_is_prior_True_indices_[3000, 6000]v12.pt')
ranks23 = torch.load(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_is_prior_True_indices_[6000, 9000]v12.pt')
ranks34 = torch.load(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_is_prior_True_indices_[9000, 12000]v12.pt')
ranks45 = torch.load(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_is_prior_True_indices_[12000, 15000]v12.pt')
ranks56 = torch.load(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_is_prior_True_indices_[15000, 18000]v12.pt')
ranks67 = torch.load(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_is_prior_True_indices_[18000, 21000]v12.pt')

# %%
ranks = torch.cat([torch.stack(ranks01, dim = 0), torch.stack(ranks12, dim = 0), torch.stack(ranks23, dim = 0), torch.stack(ranks34, dim = 0), torch.stack(ranks45, dim = 0), torch.stack(ranks56, dim = 0), torch.stack(ranks67, dim = 0)], dim = 0)
# %%
# set the max element of each row to 0
for row in ranks:
    row[row == row.max()] = 0

# %%
gene_names01 = np.load(global_variables.storage_folder + 'ppr/gene_names_0.3_indices_[0, 3000]v12.npy', allow_pickle=True)
gene_names12 = np.load(global_variables.storage_folder + 'ppr/gene_names_0.3_indices_[3000, 6000]v12.npy', allow_pickle=True)
gene_names23 = np.load(global_variables.storage_folder + 'ppr/gene_names_0.3_indices_[6000, 9000]v12.npy', allow_pickle=True)
gene_names34 = np.load(global_variables.storage_folder + 'ppr/gene_names_0.3_indices_[9000, 12000]v12.npy', allow_pickle=True)
gene_names45 = np.load(global_variables.storage_folder + 'ppr/gene_names_0.3_indices_[12000, 15000]v12.npy', allow_pickle=True)
gene_names56 = np.load(global_variables.storage_folder + 'ppr/gene_names_0.3_indices_[15000, 18000]v12.npy', allow_pickle=True)
gene_names67 = np.load(global_variables.storage_folder + 'ppr/gene_names_0.3_indices_[18000, 21000]v12.npy', allow_pickle=True)
# %%
gene_names = np.concatenate([gene_names01, gene_names12, gene_names23, gene_names34, gene_names45, gene_names56, gene_names67])
# %%
# save ranks and gene names to file
ranks = ranks.numpy()
gene_names = np.array([gene_name.lower() for gene_name in gene_names])
#np.save(global_variables.storage_folder + 'ppr/ranks_personalized_page_rank_0.3_v12.npy', ranks)
#np.save(global_variables.storage_folder + 'ppr/gene_names_0.3_v12.npy', gene_names)

# %%

# # %%

# ranks = torch.load(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/pca_components_personalized_page_rank_0.3_is_prior_True.pt')
# # # ranks = torch.stack(ranks, dim = 1)

# gene_names = np.load(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/gene_names_0.0.npy', allow_pickle=True)
# # # gene_names

# %%
# read host factors from publications
host_factors = pd.ExcelFile(global_variables.home_folder + "/data_preprocessing_pipeline/artivir_data/host_factors_from_publications.xlsx")
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

functionally_validated_host_factors = np.load(global_variables.home_folder + "data_preprocessing_pipeline/artivir_data/strong_functionally_validated_host_factors.npy")
functionally_validated_host_factors = np.array([gene.lower() for gene in functionally_validated_host_factors])

# %%
# load gene names and semantic similarity matrix from GOSemSim
gene_names = np.loadtxt(global_variables.storage_folder + 'gosemsim_matrices/gene_names_ppr_goterms_filtered.txt', dtype = str)
gene_names = np.array([gene.lower() for gene in gene_names])

df_go_sem_sim = pd.read_csv(global_variables.storage_folder + 'gosemsim_matrices/similarity_matrix_avg.csv')

# %%
for col in df_go_sem_sim.select_dtypes(include='float64').columns:
    df_go_sem_sim[col] = df_go_sem_sim[col].astype('float16')

# %%
#df_go_sem_sim.to_parquet(global_variables.storage_folder + 'gosemsim_matrices/similarity_matrix_avg_reduced.parquet')
df_go_sem_sim_reduced = pd.read_parquet(global_variables.storage_folder + 'gosemsim_matrices/similarity_matrix_avg_reduced.parquet')
go_sem_sim_reduced = df_go_sem_sim_reduced.to_numpy()
# %%
np.shape(go_sem_sim_reduced)

# %%

# read the drug targets from the file
drug_targets_df = pd.read_csv("/vol/GhostFreePro/GNN/drug_targets_df.csv")
# %%
gene_list_antiviral_new = drug_targets_df['gene_names'].values


# %%

gene_names = [gene_name.lower() for gene_name in gene_names]
indices_of_potential_host_factors = [index for index, gene_name in enumerate(gene_names) if gene_name in potential_host_factors_list]
indices_of_potential_not_host_factors = [index for index, gene_name in enumerate(gene_names) if gene_name not in potential_host_factors_list]
#indices_of_antiviral = [index for index, gene_name in enumerate(gene_names) if gene_name in gene_list_antiviral]
indices_of_antiviral = [index for index, gene_name in enumerate(gene_names) if gene_name in gene_list_antiviral_new]
indices_of_proviral = [index for index, gene_name in enumerate(gene_names) if gene_name in gene_list_proviral]
indices_of_functionally_validated = [index for index, gene_name in enumerate(gene_names) if gene_name in functionally_validated_host_factors]
# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %%
tsne = TSNE(n_components=2, perplexity=50)
tsne.fit(ranks)#.numpy())
#tsne.fit(go_sem_sim_reduced)
# %%
fig = plt.figure(figsize=(40,40))
plt.scatter(tsne.embedding_[indices_of_potential_not_host_factors, 0], tsne.embedding_[indices_of_potential_not_host_factors, 1], c = "blue", label = "unlabeled", alpha = 0.5, s = 10)
plt.scatter(tsne.embedding_[indices_of_potential_host_factors, 0], tsne.embedding_[indices_of_potential_host_factors, 1], c = "red", label = "potential host factors", alpha=0.5, s = 20)
plt.scatter(tsne.embedding_[indices_of_antiviral, 0], tsne.embedding_[indices_of_antiviral, 1], c = "black", label = "antiviral", alpha=0.99, s = 100)
plt.scatter(tsne.embedding_[indices_of_functionally_validated, 0], tsne.embedding_[indices_of_functionally_validated, 1], c = "green", label = "functionally validated", alpha=0.99, s = 60)
plt.xlabel("t-SNE 1", fontsize=30)
plt.ylabel("t-SNE 2", fontsize=30)
plt.legend(fontsize=30)
plt.title("t-SNE of Local Positional Encoding obtained via Random Walk with Restart", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# %%
# standardization: normalize each column to mean 0 and variance 1 when flow is from target to source
#ranks = ranks.numpy()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#ranks = scaler.fit_transform(ranks)
go_sem_sim_reduced = scaler.fit_transform(go_sem_sim_reduced)
#ranks = (ranks - ranks.mean(axis = 0)) / ranks.std(axis = 0)
## normalize each column to mean 0 and variance 1 when flow is from source to target
#ranks = ranks.numpy().T
#ranks = (ranks - ranks.mean(axis = 0)) / ranks.std(axis = 0)
#ranks = torch.Tensor(ranks)
# %%
pca = PCA(n_components=128)
#pca.fit(go_sem_sim_reduced)
pca.fit(ranks)

# %%
# do a scree plot to see how many components to keep
fig = plt.figure(figsize=(20,20))
plt.plot(pca.explained_variance_ratio_)
plt.xlabel("PC")
# %%
# save components to file
#torch.save(torch.tensor(pca.components_.T), global_variables.storage_folder + 'ppr/pca_components_personalized_page_rank_0.3_is_prior_True_from_target_to_source.pt')
# %%
# load components from file
pca_components_ = torch.load(global_variables.storage_folder + 'ppr/pca_components_personalized_page_rank_0.3_is_prior_True_from_target_to_source.pt')
# %%

# # print(pca.explained_variance_ratio_)
# # fig = plt.figure(figsize=(20,20))
# # plt.scatter(pca.components_[0][indices_of_potential_not_host_factors], pca.components_[1][indices_of_potential_not_host_factors], c = "blue", label = "potential negatives", alpha = 0.5, s = 1)
# # plt.scatter(pca.components_[0][indices_of_potential_host_factors], pca.components_[1][indices_of_potential_host_factors], c = "red", label = "potential host factors", alpha=0.5, s = 2)
# # plt.xlabel("PC1")
# # plt.ylabel("PC2")
# # plt.title("PCA of ranks, 0.2 threshold PPI network and with prior")
# # plt.show()


# # # %%
# # # len(pca.components_[0])

# %%
tsne = TSNE(n_components=2, perplexity=50)
tsne.fit(pca.transform(ranks))
#tsne.fit(pca.transform(go_sem_sim_reduced))
# %%
import seaborn as sns
fig = plt.figure(figsize=(40,40))
plt.scatter(tsne.embedding_[indices_of_potential_not_host_factors, 0], tsne.embedding_[indices_of_potential_not_host_factors, 1], c = "blue", label = "unlabeled", alpha = 0.5, s = 10)
plt.scatter(tsne.embedding_[indices_of_potential_host_factors, 0], tsne.embedding_[indices_of_potential_host_factors, 1], c = "green", label = "potential host factors", alpha=0.5, s = 20)
#plt.scatter(tsne.embedding_[indices_of_antiviral, 0], tsne.embedding_[indices_of_antiviral, 1], c = "black", label = "antiviral", alpha=0.99, s = 100)
#plt.scatter(tsne.embedding_[indices_of_functionally_validated, 0], tsne.embedding_[indices_of_functionally_validated, 1], c = "red", label = "functionally validated", alpha=0.99, s = 100)
#sns.kdeplot(x=tsne.embedding_[indices_of_antiviral, 0], y=tsne.embedding_[indices_of_antiviral, 1], fill=True, alpha=0.5, color='blue', levels=15, thresh=0.003)
sns.kdeplot(x=tsne.embedding_[indices_of_functionally_validated, 0], y=tsne.embedding_[indices_of_functionally_validated, 1], fill=True, alpha=0.5, color='red', levels=25, thresh=0.005)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.xlabel("t-SNE 1", fontsize=30)
plt.ylabel("t-SNE 2", fontsize=30)
plt.legend(fontsize=30)
plt.title("t-SNE of first 128 PCA components of Positional Encoding obtained via Random Walk with Restart", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# %%
import seaborn as sns
fig = plt.figure(figsize=(40,40))
plt.scatter(tsne.embedding_[indices_of_potential_not_host_factors, 0], tsne.embedding_[indices_of_potential_not_host_factors, 1], c = "blue", label = "unlabeled", alpha = 0.5, s = 10)
plt.scatter(tsne.embedding_[indices_of_potential_host_factors, 0], tsne.embedding_[indices_of_potential_host_factors, 1], c = "green", label = "potential host factors", alpha=0.5, s = 20)
#plt.scatter(tsne.embedding_[indices_of_antiviral, 0], tsne.embedding_[indices_of_antiviral, 1], c = "black", label = "antiviral", alpha=0.99, s = 100)
#plt.scatter(tsne.embedding_[indices_of_functionally_validated, 0], tsne.embedding_[indices_of_functionally_validated, 1], c = "red", label = "functionally validated", alpha=0.99, s = 100)
sns.kdeplot(x=tsne.embedding_[indices_of_antiviral, 0], y=tsne.embedding_[indices_of_antiviral, 1], fill=True, alpha=0.5, color='blue', levels=25, thresh=0.005)
#sns.kdeplot(x=tsne.embedding_[indices_of_functionally_validated, 0], y=tsne.embedding_[indices_of_functionally_validated, 1], fill=True, alpha=0.5, color='red', levels=15, thresh=0.003)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.xlabel("t-SNE 1", fontsize=30)
plt.ylabel("t-SNE 2", fontsize=30)
plt.legend(fontsize=30)
plt.title("t-SNE of first 128 PCA components of Positional Encoding obtained via Random Walk with Restart", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# %%
# # tsne_transpose = TSNE(n_components=2, perplexity=50)
# # tsne_transpose.fit(ranks.numpy().T)
# # # %%
# # fig = plt.figure(figsize=(20,20))
# # plt.scatter(tsne_transpose.embedding_[indices_of_potential_not_host_factors, 0], tsne_transpose.embedding_[indices_of_potential_not_host_factors, 1], c = "blue", label = "potential negatives", alpha = 0.5, s = 1)
# # plt.scatter(tsne_transpose.embedding_[indices_of_potential_host_factors, 0], tsne_transpose.embedding_[indices_of_potential_host_factors, 1], c = "red", label = "potential host factors", alpha=0.5, s = 2)
# # plt.xlabel("t-SNE 1")
# # plt.ylabel("t-SNE 2")
# # plt.title("t-SNE of ranks_personalized_page_rank_0.2_is_prior_False transposed rank diffuses from target to source")

# # # %%

# # # save ranks and gene names to file 
# # torch.save(ranks, global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/ranks_personalized_page_rank_0.2_is_prior_True.pt')
# # np.save(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/gene_names_ppr.npy', gene_names)

# # # %%
# # ranks
# # # %%

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Define colors
colors = ['b', 'r', 'g', 'black']

# Set the figure size
fig = plt.figure(figsize=(40, 40))

# t-SNE scatter plots
plt.scatter(tsne.embedding_[indices_of_potential_not_host_factors, 0], 
            tsne.embedding_[indices_of_potential_not_host_factors, 1], 
            c=colors[3], label="Unlabeled proteins", alpha=1.0, s=30)
plt.scatter(tsne.embedding_[indices_of_potential_host_factors, 0], 
            tsne.embedding_[indices_of_potential_host_factors, 1], 
            c=colors[2], label="Published potential host factors", alpha=0.7, s=60)

plt.scatter(tsne.embedding_[indices_of_functionally_validated, 0], 
            tsne.embedding_[indices_of_functionally_validated, 1], 
            c=colors[1], label="Functionally validated host factors", alpha=1.0, s=70)

# KDE plots from the second plot
sns.kdeplot(x=tsne.embedding_[indices_of_antiviral, 0], 
            y=tsne.embedding_[indices_of_antiviral, 1], 
            fill=False, alpha=1.0, color='blue', levels=15, thresh=0.005)
sns.kdeplot(x=tsne.embedding_[indices_of_functionally_validated, 0], 
            y=tsne.embedding_[indices_of_functionally_validated, 1], 
            fill=False, alpha=1.0, color='red', levels=35, thresh=0.005)

# Set x and y limits
plt.xlim(-80, 80)
plt.ylim(-80, 80)

# Find Arial font
arial_font = None
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'arial.' in font.lower():
        arial_font = font
        break

if arial_font is None:
    print("Arial font not found. Please ensure Arial font is installed.")
    arial_font = fm.findfont('Arial')

font_properties = fm.FontProperties(fname=arial_font)

# Set labels with Arial font and increased font size
plt.xlabel("t-SNE 1", fontsize=90, fontproperties=font_properties)
plt.ylabel("t-SNE 2", fontsize=90, fontproperties=font_properties)

# Set tick labels to Arial font and increase their font size
ax = plt.gca()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties(font_properties)
    label.set_fontsize(70)

# Title with Arial font
plt.title("t-SNE of first 128 PCA components of Positional Encoding via Random Walk with Restart", 
          fontsize=40, fontproperties=font_properties)

# Set the legend with Arial font and increased font size
plt.legend(scatterpoints=1, prop="sans\-serif:style=normal:variant=normal:weight=normal:stretch=normal:file=/usr/share/fonts/truetype/msttcorefonts/Arial.ttf:size=85.0", markerscale=4)

# Show the plot
plt.show()

# %%
