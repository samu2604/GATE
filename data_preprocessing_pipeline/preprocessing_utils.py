import numpy as np
import pandas as pd
import os.path
import h5py  
import pathlib
import tqdm
import random
from datetime import datetime
from functools import reduce
from GNN.global_variables import global_vars as global_variables

global data_file_path

def create_hdf5_container(network, features, node_names, feat_names, y_train, train_mask, y_test, test_mask, y_val, val_mask, container_name :str, container_data_path):
    
    f = h5py.File(pathlib.Path(container_data_path, container_name + '.h5'), 'w')
    string_dt = h5py.special_dtype(vlen=str)
    f.create_dataset('network', data=network, shape=network.shape)
    f.create_dataset('features', data=features, shape=features.shape)
    f.create_dataset('gene_names', data=node_names, dtype=string_dt)
    f.create_dataset('y_train', data=y_train, shape=y_train.shape)
    f.create_dataset('y_val', data=y_val, shape=y_val.shape)
    f.create_dataset('y_test', data=y_test, shape=y_test.shape)
    f.create_dataset('mask_train', data=train_mask, shape=train_mask.shape)
    f.create_dataset('mask_val', data=val_mask, shape=val_mask.shape)
    f.create_dataset('mask_test', data=test_mask, shape=test_mask.shape)
    f.create_dataset('feature_names', data=feat_names, dtype=string_dt)
    f.close()

def create_network(df_ppi: pd.DataFrame, randomized_gene_list: list) -> list:
    # create a lookup dictionary to get indices from gene names
    gene_indices_dict = {gene_name: index for index, gene_name in enumerate(randomized_gene_list)}
    gene_name_1_array = df_ppi["gene_name_1"].values
    gene_name_2_array = df_ppi["gene_name_2"].values

    combined_score_array = df_ppi["combined_score_recomputed"].values
    edge_index_0 = []
    edge_index_1 = []
    edge_weight = []
    for gene_1, gene_2, score in tqdm.tqdm(zip(gene_name_1_array, gene_name_2_array, combined_score_array)):
        edge_index_0.append(gene_indices_dict[gene_1])
        edge_index_1.append(gene_indices_dict[gene_2])
        edge_weight.append(score)
    network = [edge_index_0, edge_index_1, edge_weight]
    return network

def create_adjacency_matrix_and_feature_matrix(df_ppi, features_dict, omic_data_type, is_debug, is_dataset_emogi_compatible, interaction_score_threshold = 0.7):
    # let's create the list of genes for which we have features
    gene_list = [gene_name for gene_name in features_dict.keys()]
    list_of_connected_genes = global_variables.connected_genes
    list_of_connected_genes = [gene for gene in list_of_connected_genes if gene in gene_list]

    list_of_connected_genes_set = set(list_of_connected_genes)
    mask_gene_1 = df_ppi["gene_name_1"].isin(list_of_connected_genes_set)
    mask_gene_2 = df_ppi["gene_name_2"].isin(list_of_connected_genes_set)
    print("Selecting only genes with features and dropping the rest")
    df_ppi = df_ppi[mask_gene_1 & mask_gene_2]
    df_ppi.reset_index(drop = True, inplace = True)
    
    new_dir_name = 'input_data'
    feature_vector_length = 0
    if "transcriptome" in omic_data_type:
        new_dir_name += "_transcriptome"
        feature_vector_length += 3
    if "proteome" in omic_data_type:
        new_dir_name += "_proteome"
        feature_vector_length += 3
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        if "effectome" in omic_data_type:
            new_dir_name += "_effectome"
            feature_vector_length += 24
        if "interactome" in omic_data_type:
            new_dir_name += "_interactome"
            feature_vector_length += 24     

    if is_dataset_emogi_compatible:
        date_time_obj = datetime.now()
        new_dir_name = new_dir_name + "_" + str(date_time_obj.year) + "_" + str(date_time_obj.month) + "_" + str(date_time_obj.day) + "_" + str(date_time_obj.hour) + "_" + str(date_time_obj.minute) 
        new_dir = pathlib.Path(data_file_path, new_dir_name)
        new_dir.mkdir(parents=True, exist_ok=True)

    max_interaction_strength = np.max(df_ppi["combined_score"].values)

        # compute the adjacency matrix
    print("Create adjacency matrix  ")
    if is_dataset_emogi_compatible:
        # compute the randomized gene list    
        print("Create randomized gene list")
        gene_indices_list =  np.random.permutation(np.arange(0, len(gene_list)))
        randomized_gene_list = [gene_list[index] for index in gene_indices_list]   
        # create a lookup dictionary to get indices from gene names
        gene_indices_dict = {gene_name: index for index, gene_name in enumerate(randomized_gene_list)}
        print("Dataset is emogi compatible")
        network = np.zeros([len(randomized_gene_list) ,len(randomized_gene_list)])    
        for gene in tqdm.tqdm(list_of_connected_genes):
            genes_connected_to_gene = df_ppi[df_ppi["gene_name_1"] == gene]["gene_name_2"].values
            # some of the genes in column "gene_name_2" are not in the featured gene list for the multiscale interactome
            connected_to_gene_interaction_strength_genes = df_ppi[df_ppi["gene_name_1"] == gene]["combined_score"].values
            index_gene = gene_indices_dict[gene]
            for gene_connected, gene_connected_interaction_strength in zip(genes_connected_to_gene, connected_to_gene_interaction_strength_genes):     
                if gene_connected in randomized_gene_list:  
                    index_gene_connected_to_gene = gene_indices_dict[gene_connected]
                    if network[index_gene, index_gene_connected_to_gene] == 0 and index_gene != index_gene_connected_to_gene: 
                        network[index_gene, index_gene_connected_to_gene] = gene_connected_interaction_strength/max_interaction_strength 
                        network[index_gene_connected_to_gene, index_gene] = gene_connected_interaction_strength/max_interaction_strength    
    else:
        print("Dataset is not emogi compatible")
        new_dir = ""
        #df_ppi["combined_score_recomputed"] = df_ppi["combined_score_recomputed"]#* (1.0/max_interaction_strength)
        #df_ppi["combined_score_recomputed"] = df_ppi["combined_score_recomputed"]**2
        # select only the connections with score above the threshold
        df_ppi = df_ppi[df_ppi["combined_score_recomputed"] > interaction_score_threshold]
        df_ppi.reset_index(drop = True, inplace = True)
        print("The number of connections is " + str(len(df_ppi)))
        randomized_gene_list = df_ppi["gene_name_1"].unique() #this operation preserves the original order, it can also be randomized with np.random.permutation
        if is_debug:
            randomized_gene_list = np.unique(np.concatenate((randomized_gene_list, df_ppi["gene_name_2"].unique()), axis = 0))

        network = create_network(df_ppi, randomized_gene_list)

    # save the network and the randomized gene list in the new dedicated folder
    if is_dataset_emogi_compatible:
        np.savetxt(new_dir / "randomized_gene_list.txt", randomized_gene_list, delimiter = " ", fmt="%s")
        np.savez_compressed(new_dir /"network", network, network=network)
    #save copy for debug purposes
    if not is_debug and is_dataset_emogi_compatible:
        np.savetxt(data_file_path +  "/randomized_gene_list_tmp.txt", randomized_gene_list, delimiter = " ", fmt="%s")
        np.savez_compressed(data_file_path /"network", network, network=network)
        

    # compute the fatures matrix
    features = np.zeros([len(randomized_gene_list) ,feature_vector_length])
    node_names = []
    print("Feature matrix creation")
    for gene_index, gene_name in enumerate(randomized_gene_list):
        node_names.append([gene_name, gene_name])
        feature_index = 0
        if "transcriptome" in omic_data_type:
            for i in range(0,3):
                features[gene_index, feature_index] = features_dict[gene_name][i] # transcriptomics
                feature_index += 1
        if "proteome" in omic_data_type:
            for i in range(3,6):    
                features[gene_index, feature_index] = features_dict[gene_name][i] # proteomics 
                feature_index += 1
        if global_variables.fine_tuning_virus == "SARS-CoV-2":
            if "effectome" in omic_data_type:
                for i in range(6,30):
                    features[gene_index, feature_index] = features_dict[gene_name][i] # effectome
                    feature_index += 1    
            if "interactome" in omic_data_type:
                for i in range(30,54):
                    features[gene_index, feature_index] = features_dict[gene_name][i] # interactome
                    feature_index += 1  
    if is_dataset_emogi_compatible:                      
        np.savetxt(new_dir / "features.csv", features, delimiter = ",") 
    node_names = np.array(node_names, dtype=object)                
    
    feat_names = np.array([], dtype=object)
    if "transcriptome" in omic_data_type:
        feat_names = np.array(np.concatenate((feat_names, ["transcriptomics_06h", "transcriptomics_12h", "transcriptomics_24h"]), axis=0), dtype=object)
    if "proteome" in omic_data_type:   
        feat_names = np.array(np.concatenate((feat_names, ["proteomics_06h", "proteomics_12h" ,"proteomics_24h"]), axis=0), dtype=object) 
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        if "effectome" in omic_data_type:    
            feat_names = np.array(np.concatenate((feat_names, global_variables.viral_proteins), axis=0), dtype=object)
        if "interactome" in omic_data_type:    
            feat_names = np.array(np.concatenate((feat_names, global_variables.viral_proteins), axis=0), dtype=object)        
                     
    return network, features, node_names, feat_names, new_dir, randomized_gene_list

    
def sample_from_list_and_return_remaining_list(list_to_sample, number_elements_to_sample):
    if number_elements_to_sample > len(list_to_sample):
        print("Number of elements to sample is bigger then the sampling list")
    sampled_elements = random.sample(list_to_sample, number_elements_to_sample)
    for element in sampled_elements:
       list_to_sample.remove(element)
                
    return sampled_elements, list_to_sample
    
def train_test_val_sampler(list_to_sample, number_train_samples, number_test_samples, number_val_samples):
    training_samples, list_to_sample = sample_from_list_and_return_remaining_list(list_to_sample, number_train_samples)
    test_samples, list_to_sample = sample_from_list_and_return_remaining_list(list_to_sample, number_test_samples)
    validation_samples, list_to_sample = sample_from_list_and_return_remaining_list(list_to_sample, number_val_samples)
    return training_samples, test_samples, validation_samples

def read_ppi_data_and_map_to_gene_names(path_to_string: str, path_to_uniprot: str = global_variables.storage_folder + "HUMAN_9606_idmapping.dat"):
   
    if not os.path.exists(os.path.join(os.path.dirname(path_to_string), 'string_to_gene_names_map.csv')):
        print(f"load uniprot and map the content to STRING ensemble ids and save it in the same folder of {path_to_string}")
        uniprot_dat_file_content = [i.strip().split() for i in open(path_to_uniprot).readlines()]
        string_and_gene_names = []
        for item in uniprot_dat_file_content:
            if (item[1] == "Gene_Name") or (item[1] == "STRING"):
                string_and_gene_names.append(item) 
        print("Create a map from STRING protein ids to gene names")
        string_names_list = []
        gene_names_list = []
        for elem in tqdm.tqdm(string_and_gene_names, desc="look for STRING protein ids corresponding to gene names in dataset"):
            if elem[1] == "STRING":
                for names in string_and_gene_names: # check if the uniprot name is the same
                    if names[0] == elem[0] and names[1] == "Gene_Name":
                        string_names_list.append(elem[2])
                        gene_names_list.append(names[2])
                        break     
                    
        string_to_gen_names_map = {"string_ids" : string_names_list, "gene_names" : gene_names_list}        
        df_string_to_gen_names_map = pd.DataFrame(string_to_gen_names_map)
        df_string_to_gen_names_map.set_index('string_ids', inplace=True)

        # save df_string_to_gen_names_map to file for future use in the same folder of path_to_string
        df_string_to_gen_names_map.to_csv(os.path.join(os.path.dirname(path_to_string), 'string_to_gene_names_map.csv'))
    else:
        print("Load map from STRING protein ids to gene names")
        df_string_to_gen_names_map = pd.read_csv(os.path.join(os.path.dirname(path_to_string), 'string_to_gene_names_map.csv'), index_col=0)

    df_string = pd.read_csv(path_to_string, sep=' ', header=0)

    print("Map protein ENS Ids to gene names")
    df_merged_1 = pd.merge(df_string, df_string_to_gen_names_map, left_on='protein1', right_index=True, how='left')
    df_merged_1.rename(columns={'gene_names': 'gene_name_1'}, inplace=True)

    df_merged_2 = pd.merge(df_string, df_string_to_gen_names_map, left_on='protein2', right_index=True, how='left')
    df_merged_2.rename(columns={'gene_names': 'gene_name_2'}, inplace=True)

    df_string['gene_name_1'] = df_merged_1['gene_name_1']
    df_string['gene_name_2'] = df_merged_2['gene_name_2']

    df_string.fillna('nan', inplace=True)

    return df_string

# function that aggregates the set of scores in string ppi according to a set of weights that are passed as input

def aggregate_scores(path_to_string: str, weights: list = None, scores_to_concatenate: list = ['neighborhood', 'neighborhood_transferred', 
                                                                                               'fusion', 'cooccurence', 'coexpression', 
                                                                                               'coexpression_transferred', 'experiments',
                                                                                               'experiments_transferred', 'database', 
                                                                                               'database_transferred', 'textmining',
                                                                                               'textmining_transferred']) -> pd.core.frame.DataFrame:
    
    # load the ppi
    df_ppi = read_ppi_data_and_map_to_gene_names(path_to_string)
    # convert scores in probabilities
    if weights is None:
        weights = [1,1,1,1,1,1,1,1,1,1,1,1]   
    try :
        assert len(weights) == len(scores_to_concatenate)
        pass
    except:
        print("The number of weights is not correct")
        return None
    
    prior = 0.041
    df_ppi["homology"] = df_ppi["homology"]/1000
    for col, weight in zip(scores_to_concatenate, weights):
        df_ppi[col] = df_ppi[col]/1000
        df_ppi[col + "_no_prior"] = (df_ppi[col] - prior)/(1-prior)
        df_ppi.loc[df_ppi[col + "_no_prior"] < 0, col + "_no_prior"] = 0.0
        if col == "cooccurence" or col == "textmining" or col == "textmining_transferred":
            df_ppi[col + "_no_prior"] = df_ppi[col + "_no_prior"]*(1 - df_ppi["homology"])
        df_ppi[col + "_no_prior"] = df_ppi[col + "_no_prior"]*weight
    
    df_ppi["combined_score_no_prior"] = 1 - (1 - df_ppi["neighborhood_no_prior"])*(1 - df_ppi["neighborhood_transferred_no_prior"])*(1 - df_ppi["fusion_no_prior"])\
        *(1 - df_ppi["cooccurence_no_prior"])*(1 - df_ppi["coexpression_no_prior"])*(1 - df_ppi["coexpression_transferred_no_prior"])*(1 - df_ppi["experiments_no_prior"])\
            *(1 - df_ppi["experiments_transferred_no_prior"])*(1 - df_ppi["database_no_prior"])*(1 - df_ppi["database_transferred_no_prior"])\
                *(1 - df_ppi["textmining_no_prior"])*(1 - df_ppi["textmining_transferred_no_prior"])
    df_ppi["combined_score_recomputed"] = df_ppi["combined_score_no_prior"]*(1-prior) + prior

    return df_ppi
        
        
