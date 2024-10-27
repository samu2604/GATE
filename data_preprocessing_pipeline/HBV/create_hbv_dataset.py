from statistics import mean, stdev, median
import pandas as pd
import numpy as np
import argparse
import os.path
from tqdm import tqdm
from sklearn.decomposition import PCA
from datetime import datetime
import pathlib
import sys, os
sys.path.append(os.path.abspath(str(pathlib.Path.home()) + '/GhostFreePro/data_preprocessing_pipeline/'))
import importlib
from preprocessing_utils import create_hdf5_container, train_test_val_sampler
from feature_preselection_transcriptome_proteome import map_ensp_to_gene_name, add_gene_names_to_ppi, train_test_val_split, extract_feature_from_row
importlib.reload(sys.modules['preprocessing_utils'])
importlib.reload(sys.modules['feature_preselection_transcriptome_proteome'])
hbv_data_path = "/home/samuele/GhostFreePro/data_preprocessing_pipeline/HBV/"
transcriptome_file = "/home/samuele/GhostFreePro/data_preprocessing_pipeline/HBV/Lucko RNA Seq Data.xlsx"

def find_detected_genes(transcriptome_file: str):
    transcriptomics_data = pd.ExcelFile(transcriptome_file)
    df_transcriptomics = transcriptomics_data.parse("Sheet1")
    df_transcriptomics = df_transcriptomics.dropna(subset=["SYMBOL", "logFC", "adj.P.Val"])
    transcriptomics_genes = df_transcriptomics["SYMBOL"].unique()
    return df_transcriptomics

# create a function that filters the ppi network based on the genes that are detected in the transcriptomics data
def ppi_prep(transcriptomics_genes: np.ndarray):
    if os.path.isfile(hbv_data_path + "df_string_ppi_complete.zip"):
        df_string_ppi = pd.read_csv(hbv_data_path + "df_string_ppi_complete.zip")
        return df_string_ppi
    df_string_ppi = pd.read_csv(hbv_data_path + "df_string_ppi_with_gene_names.zip")
    df_string_ppi["gene_name_1"] = df_string_ppi["gene_name_1"].str.lower()
    df_string_ppi["gene_name_2"] = df_string_ppi["gene_name_2"].str.lower()
    transcriptomics_genes = np.array([s.lower() for s in transcriptomics_genes])
    list_of_row_indices_to_drop = []
    row_number = df_string_ppi.shape[0]
    for index, row in tqdm(df_string_ppi.iterrows(), desc=f"Finding genes not detected at all by transcriptomics",\
                            total=row_number):
        if (row["gene_name_1"] not in transcriptomics_genes) or (row["gene_name_2"] not in transcriptomics_genes):
            list_of_row_indices_to_drop.append(index)
    df_string_ppi.drop(list_of_row_indices_to_drop, axis = 0, inplace = True)            
    df_string_ppi.reset_index(drop = True, inplace = True)
    #df_string_ppi.to_csv(hbv_data_path + 'df_string_ppi_complete.zip', index=False, compression = dict(method='zip',archive_name='df_string_ppi_complete.csv'))
    return df_string_ppi    

def trascriptomics_features_prep(df_transcriptomics, transcriptomics_genes, df_ppi):
    connected_common_genes = []
    connected_genes = np.unique(np.concatenate((df_ppi['gene_name_1'].unique(),df_ppi['gene_name_2'].unique()), axis=0))
    for gene in tqdm(transcriptomics_genes, desc="Selecting connected genes among genes detected in the data"):
        if gene in connected_genes:
            connected_common_genes.append(gene)
        
    feature_dict = {}
    timestamp_list_transcriptomics=[""] 
    # bring all the gene names to lower case     
    df_transcriptomics["SYMBOL"] = df_transcriptomics["SYMBOL"].str.lower()
    for gene in tqdm(connected_common_genes, desc="Assigning features to connected genes"):
        feature_vector = []
        row = df_transcriptomics[df_transcriptomics["SYMBOL"] == gene]
        if row.shape[0] > 0:
            extract_feature_from_row(row, timestamp_list_transcriptomics, feature_vector, is_used_abs_value_for_up_down_regulated=False, \
                                      fold_change_first_part="logFC", p_val_first_part="adj.P.Val")
            
        feature_dict.update({gene : feature_vector})
            
    np.save(hbv_data_path + 'feature_dict_tmp.npy', feature_dict)
             
    return feature_dict     

def create_adjacency_matrix_and_feature_matrix(df_ppi, features_dict):
    gene_list = [gene_name for gene_name in features_dict.keys()]
    list_of_connected_genes = np.unique(np.concatenate((df_ppi["gene_name_1"].values, df_ppi["gene_name_2"].values), axis=0))
    list_of_connected_genes = [gene for gene in list_of_connected_genes if gene in gene_list]
    
    max_interaction_strength = max(df_ppi["experiments"].values)
    new_dir_name = 'input_data'
    feature_vector_length = 0
    new_dir_name += "_transcriptome"
    feature_vector_length += 1
      
                   
    date_time_obj = datetime.now()
    new_dir_name = new_dir_name + "_" + str(date_time_obj.year) + "_" + str(date_time_obj.month) + "_" \
          + str(date_time_obj.day) + "_" + str(date_time_obj.hour) + "_" + str(date_time_obj.minute) 

    new_dir = pathlib.Path(hbv_data_path, new_dir_name)
    new_dir.mkdir(parents=True, exist_ok=True) 

    # compute the randomized gene list    
    print("Create randomized gene list")
    gene_indices_list =  np.random.permutation(np.arange(0, len(gene_list)))
    randomized_gene_list = [gene_list[index] for index in gene_indices_list]   
    # compute the adjacency matrix
    print("Create adjacency matrix  ")
    network = np.zeros([len(randomized_gene_list) ,len(randomized_gene_list)])   
    for gene in tqdm(list_of_connected_genes):
        genes_connected_to_gene = df_ppi[df_ppi["gene_name_1"] == gene]["gene_name_2"].values
        # some of the genes in column "gene_name_2" are not in the featured gene list for the multiscale interactome
        connected_to_gene_interaction_strength_genes = df_ppi[df_ppi["gene_name_1"] == gene]["experiments"].values
        index_gene = randomized_gene_list.index(gene)
        for gene_connected, gene_connected_interaction_strength in zip(genes_connected_to_gene, connected_to_gene_interaction_strength_genes):     
            if gene_connected in randomized_gene_list:  
                index_gene_connected_to_gene = randomized_gene_list.index(gene_connected)
                if network[index_gene, index_gene_connected_to_gene] == 0 and index_gene != index_gene_connected_to_gene: 
                    network[index_gene, index_gene_connected_to_gene] = (gene_connected_interaction_strength/max_interaction_strength)**2 
                    network[index_gene_connected_to_gene, index_gene] = (gene_connected_interaction_strength/max_interaction_strength)**2   
                    
    np.savetxt(new_dir / "randomized_gene_list.txt", randomized_gene_list, delimiter = " ", fmt="%s")
    np.savez_compressed(new_dir /"network", network, network=network) 
        
    # compute the fatures matrix
    features = np.zeros([len(randomized_gene_list) ,feature_vector_length])
    node_names = []
    print("Feature matrix creation")
    for gene_index, gene_name in enumerate(randomized_gene_list):
        node_names.append([gene_name, gene_name])
        feature_index = 0
        features[gene_index, feature_index] = features_dict[gene_name][0] # transcriptomics feature
        feature_index += 1
    np.savetxt(new_dir / "features.csv", features, delimiter = ",") 
    node_names = np.array(node_names, dtype=object)                
    
    feat_names = np.array(["transcriptomics"], dtype=object)    
                     
    return network, features, node_names, feat_names, new_dir

def extract_positives_and_potential_positives(ranked_genes, host_factors):
    host_factors = host_factors.parse()["Gene Symbol"].str.lower().unique()
    ranked_genes = ranked_genes.parse()["id"].str.lower().values

    def is_float(value):
        return isinstance(value, float)

    is_float_vec = np.vectorize(is_float)
    float_mask = is_float_vec(ranked_genes)
    # filter out the genes that are listed as "nan" in the ranked genes file
    ranked_genes = ranked_genes[~float_mask]
    potential_negative_genes = ranked_genes[15000:]
    potential_positive_genes = ranked_genes[:15000]

    positives = []
    for gene in host_factors:
        if gene not in potential_negative_genes:
            positives.append(gene)
    positives = np.array(positives)        
    positives = np.unique(np.concatenate((positives, host_factors[:50]), axis=0))
    return positives, potential_positive_genes, potential_negative_genes

if __name__=="__main__":
    map_ensp_to_gene_name(uniprot_file_path = hbv_data_path + "HUMAN_9606_idmapping.dat",  data_path=hbv_data_path)
    add_gene_names_to_ppi(data_path=hbv_data_path)

    df_transcriptomics = find_detected_genes(transcriptome_file)
    transcriptomics_genes = df_transcriptomics["SYMBOL"].str.lower().unique()
    df_string_ppi = ppi_prep(transcriptomics_genes)
    # filter the connection with zero experiments score
    df_string_ppi = df_string_ppi[df_string_ppi["experiments"] > 0].reset_index()
    feature_dict = trascriptomics_features_prep(df_transcriptomics, transcriptomics_genes, df_string_ppi)

    network, features, node_names, feat_names, new_dir = create_adjacency_matrix_and_feature_matrix(df_string_ppi, feature_dict)

    host_factors = pd.ExcelFile(hbv_data_path + "HBV essential genes list.xlsx")
    ranked_genes = pd.ExcelFile(hbv_data_path + "mageck_HBVCre_Screening_selected-VS-Mock.gene_summary.xlsx")
    positives, potential_positive_genes, potential_negative_genes = extract_positives_and_potential_positives(ranked_genes, host_factors)

    y_train, train_mask, y_test, test_mask, y_val, val_mask = train_test_val_split(new_dir, potential_positive_genes,
                                                                                   positives, "", 
                                                                                   are_stukalov_shared_genes_used = False,
                                                                                   is_select_all_potential_negatives = True)

    create_hdf5_container(network, features, node_names, feat_names, y_train, train_mask, y_test, test_mask, y_val, val_mask, "transcriptomics_HBV", new_dir)
    print(pathlib.Path(new_dir, "transcriptomics_HBV" + '.h5'))  

# testing code
# %%

# positives_list = host_factors.parse()["Gene Symbol"].str.lower().unique()
# ranks_list = ranked_genes.parse()["id"].str.lower().values

# positives_ranked = []
# for gene in positives_list:
#     if gene in ranks_list:
#         positives_ranked.append(np.where(ranks_list == gene)[0][0])
# # %%
# from matplotlib import pyplot as plt
# plt.xlabel("positive rank")
# plt.ylabel("counts")
# plt.hist(positives_ranked, bins=100)

# y_tot = (y_train + y_test + y_val)[:,0]
# dataset_genes = np.genfromtxt(new_dir / "randomized_gene_list.txt", dtype=str)
# dataset_pos = dataset_genes[y_tot.astype(bool)]
# # %%
# for gene in dataset_pos:
#     if gene in positives:
#         print(gene)
# # %%
# counter = 0
# dataset_neg = dataset_genes[((train_mask + val_mask + test_mask) - y_tot).astype(bool)] 
# for gene in dataset_neg:
#     if gene not in potential_negative_genes:
#         counter += 1

# # %%
# from matplotlib import pyplot as plt
# connection_strength_values = network[np.where(network != 0)[0], np.where(network != 0)[1]]
# plt.hist(connection_strength_values, bins=40)

# # %%
# counter = 0
# counter_2 = 0
# for gene in potential_positive_genes:
#     if gene not in node_names[:,0]:
#         counter += 1
#         if gene in potential_negative_genes:
#             counter_2 += 1
# # %%
# df_string_ppi = df_string_ppi.rename(columns={"combined_score": "experiments"})

# # %%
# len(df_transcriptomics["SYMBOL"].unique())
# # %%
# string_lower = df_string_ppi["gene_name_1"].str.lower()
# # %%
# counter = 0
# for name in df_string_ppi["gene_name_1"].str.lower().unique():
#     if name not in df_transcriptomics["SYMBOL"].str.lower().unique():
#         counter += 1

# # %%
# counter = 0
# for elem in ranked_genes:
#     if type(elem) != str:
#         counter +=1
#         print(elem) 
# # %%
