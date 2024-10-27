from statistics import mean, stdev, median
import pandas as pd
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join('/vol/storage/GhostFreePro')))
from GNN.global_variables import global_vars as global_variables
import os.path
from tqdm import tqdm
from sklearn.decomposition import PCA
import random  
import pickle
from data_preprocessing_pipeline.preprocessing_utils import create_hdf5_container, create_network, create_adjacency_matrix_and_feature_matrix, train_test_val_sampler, aggregate_scores
import torch 
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

artivir_data_path = global_variables.home_folder + "data_preprocessing_pipeline/artivir_data/"
storage_folder = global_variables.storage_folder


debug = {"multiomics_ppi_prep": True, "feature_dict_prep" : False, "adjacency_matrix_prep": True, "simple_debug": global_variables.is_simple_debug}
is_dataset_emogi_compatible = global_variables.is_dataset_emogi_compatible

## %%
# # select gene_names_1 and protein1 columns from df_string_ppi
# #df_string_ppi_filtered = df_string_ppi[["protein1", "protein2", "gene_name_1", "gene_name_2"]]
# # %%

# # %%
# df_string_ppi = pd.read_csv(artivir_data_path + "df_string_ppi_complete.zip")
# map_gene_name_to_ensembl_id = pd.read_csv(artivir_data_path + "unique_detected_gene_names.annot.txt", sep = "\t")

# map_gene_name_to_ensembl_id = map_gene_name_to_ensembl_id.rename(columns={"alias": "gene_name_1", "#string_protein_id": "protein1"})
# df_string_ppi = df_string_ppi.merge(map_gene_name_to_ensembl_id, on="protein1", how="left")
# df_string_ppi["gene_name_1_y"] = df_string_ppi["gene_name_1_y"].str.lower()
# df_string_ppi["gene_name_1_y"] = df_string_ppi["gene_name_1_y"].fillna(df_string_ppi["gene_name_1_x"])
# df_string_ppi = df_string_ppi.rename(columns={"gene_name_1_x": "gene_name_1_old"})
# df_string_ppi = df_string_ppi.rename(columns={"gene_name_1_y": "gene_name_1"})

# map_gene_name_to_ensembl_id = map_gene_name_to_ensembl_id.rename(columns={"gene_name_1": "gene_name_2", "protein1": "protein2"})
# df_string_ppi = df_string_ppi.merge(map_gene_name_to_ensembl_id, on="protein2", how="left")
# df_string_ppi["gene_name_2_y"] = df_string_ppi["gene_name_2_y"].str.lower()
# df_string_ppi["gene_name_2_y"] = df_string_ppi["gene_name_2_y"].fillna(df_string_ppi["gene_name_2_x"])
# df_string_ppi = df_string_ppi.rename(columns={"gene_name_2_x": "gene_name_2_old"})
# df_string_ppi = df_string_ppi.rename(columns={"gene_name_2_y": "gene_name_2"})
# # %%
# # save remapped ppi network to file
# df_string_ppi.to_csv(artivir_data_path + "df_string_ppi_complete.zip", index=False)
# # %%

def create_trivial_features(features, y_train, train_mask, y_test, test_mask, y_val, val_mask, hdf5_file_name):
    hdf5_file_name += "_trivial_features"
    
    positive_labelled_indices = np.unique(np.concatenate((np.where(y_val > 0.1)[0], np.where(y_train > 0.1)[0], np.where(y_test > 0.1)[0]), axis = None))
    
    all_dataset_indices = np.unique(np.concatenate((np.where(train_mask > 0)[0], np.where(test_mask > 0)[0], np.where(val_mask > 0)[0]), axis = None))  
    
    remaining_indices = []
    for index in np.arange(0, len(features)):
        if index not in all_dataset_indices:
            remaining_indices.append(index)  
                    
    remaining_indices = np.random.permutation(np.array(remaining_indices)) 
    remaining_positive_indices = remaining_indices[: int(len(remaining_indices)/4)]
    
    _, n_features = np.shape(features)
        
    for index in positive_labelled_indices:
        for feature in np.arange(0, n_features):
            features[index][feature] = np.random.normal(loc=1.0, scale=0.15, size=None)
    
    for index in remaining_positive_indices:
        for feature in np.arange(0, n_features):
            features[index][feature] = np.random.normal(loc=1.0, scale=0.15, size=None)        
            
    for index in np.arange(0, len(features)):
        if index not in positive_labelled_indices and index not in remaining_positive_indices:
            for feature in np.arange(0, n_features):
                features[index][feature] = np.random.normal(loc=-1.0, scale=0.15, size=None)           
        
    return features, hdf5_file_name
 
def compute_imputed_positives_percentage(positive_labels, genes_with_imputed_feature_values):
    total_positives = len(positive_labels)  

    number_of_imputed_genes = 0   
    for gene in positive_labels:    
        if gene in genes_with_imputed_feature_values:
            number_of_imputed_genes += 1
            
    return number_of_imputed_genes/total_positives   

def make_positives_and_negatives_distributions_similar(potential_negative_labels, imputed_genes, imputed_positives_ratio):     
    imputed_potential_negative_labels = []
    not_imputed_potential_negative_labels = []
    for gene in potential_negative_labels:
        if gene in imputed_genes:
           imputed_potential_negative_labels.append(gene)
        else:
           not_imputed_potential_negative_labels.append(gene)    
    
    number_imputed_negative_labels_to_extract = int(len(not_imputed_potential_negative_labels)*(imputed_positives_ratio/(1 - imputed_positives_ratio)))
    extracted_imputed_negative_labels = random.sample(imputed_potential_negative_labels, number_imputed_negative_labels_to_extract)
    
    return np.concatenate((not_imputed_potential_negative_labels, extracted_imputed_negative_labels))
              
    
def train_test_val_split(node_gene_list: np.ndarray, potential_host_factors_from_publications, strong_host_factors,
                         host_factors_shared_from_stukalov_and_others :str, 
                         validation_set_genes_path: str, 
                         are_stukalov_shared_genes_used: bool,
                         disease_genes_related_to_viral_infections: str=None,
                         use_all_potential_negatives: bool=False,
                         use_subset_of_negatives=True,
                         fraction_of_negatives_to_select: float=0.3,
                         is_virus_specific_model: bool=True,
                         is_pretraining_split: bool=False):
    
    nodes_number = len(node_gene_list)
    y_train = np.zeros([nodes_number, 1], float)
    y_test = np.zeros([nodes_number, 1], float)
    y_val = np.zeros([nodes_number, 1], float)
    train_mask = np.zeros(nodes_number, float)
    test_mask = np.zeros(nodes_number, float)
    val_mask = np.zeros(nodes_number, float)  

    if global_variables.fine_tuning_virus == "SARS-CoV-2" and not is_pretraining_split:
        # Read positive labels
        if isinstance(strong_host_factors, str):
            positive_host_factors = np.load(strong_host_factors)
        elif isinstance(strong_host_factors, np.ndarray):
            positive_host_factors = strong_host_factors

        if are_stukalov_shared_genes_used:
            # Read shared host factors from stukalov  
            host_factors_shared_from_stukalov_and_others = np.genfromtxt(host_factors_shared_from_stukalov_and_others, dtype=str)  

            # join positives
            positive_host_factors = np.unique(np.concatenate((positive_host_factors, host_factors_shared_from_stukalov_and_others), axis = None))
        positive_host_factors = np.char.lower(positive_host_factors.astype(str)) 

        if isinstance(potential_host_factors_from_publications, str):
            host_factors = pd.ExcelFile(potential_host_factors_from_publications)
            host_factors_dfs = {sheet_name: host_factors.parse(sheet_name) for sheet_name in host_factors.sheet_names}
            potential_host_factors_to_remove = host_factors_dfs["host_factors"]["Gene name"].unique() 
        elif isinstance(potential_host_factors_from_publications, np.ndarray):
            potential_host_factors_to_remove = potential_host_factors_from_publications
    elif global_variables.fine_tuning_virus == "MPXV" and not is_pretraining_split:
        positive_host_factors = global_variables.host_factors_mpxv
        potential_host_factors_to_remove = global_variables.potential_positive_genes_mpxv
    elif is_pretraining_split:
        ValueError("The strong host factors should be a numpy array") if not isinstance(strong_host_factors, np.ndarray) else None
        ValueError("The potential host factors from publications should be a numpy array") if not isinstance(potential_host_factors_from_publications, np.ndarray) else None
        positive_host_factors = strong_host_factors
        potential_host_factors_to_remove = potential_host_factors_from_publications

    potential_host_factors_to_remove = np.char.lower(potential_host_factors_to_remove.astype(str)) 
    if isinstance(disease_genes_related_to_viral_infections, str):
        disease_genes_related_to_viral_infections = np.genfromtxt(disease_genes_related_to_viral_infections, dtype=str)
        potential_host_factors_to_remove = np.unique(np.concatenate((potential_host_factors_to_remove, disease_genes_related_to_viral_infections), axis = None))

    potential_host_factors_to_remove = np.char.lower(potential_host_factors_to_remove.astype(str))   
    print(f"Number of potential host factors to remove: {len(potential_host_factors_to_remove)}")
   
    
    # select and split positive labels in my dataset
    positive_labels = []
    for gene in tqdm(positive_host_factors, desc="select and split positive labels in my dataset"):
        if gene in node_gene_list:
            positive_labels.append(gene)
    
    number_positives = len(positive_labels)
    #imputed_positives_ratio = compute_imputed_positives_percentage(positive_labels, genes_with_imputed_feature_values)
    #print(f"The ratio of imputed positives is: {imputed_positives_ratio}")
    train_set_pos = int(number_positives * 0.70)
    test_set_pos = int(number_positives * 0.20)
    val_set_pos = int(number_positives * 0.1) + (number_positives - test_set_pos - train_set_pos - int(number_positives * 0.1))       
        
    validation_set_genes = []
    if is_virus_specific_model:
        # training virus specific model
        validation_set_genes = np.genfromtxt(validation_set_genes_path, dtype=str)
        validation_set_genes = np.char.lower(validation_set_genes.astype(str))
    # select the potential negatives in my dataset
    potential_negative_labels = []     
    for gene in tqdm(node_gene_list, desc="select the potential negatives in my dataset"):
        if gene not in potential_host_factors_to_remove and gene not in positive_labels and gene not in validation_set_genes:
            potential_negative_labels.append(gene)
    
    #potential_negative_labels = make_positives_and_negatives_distributions_similar(potential_negative_labels, genes_with_imputed_feature_values, imputed_positives_ratio)        
    #potential_negative_labels = potential_negative_labels.tolist()
    print(f"Length of potential_negative_labels {len(potential_negative_labels)} ") 
    # shuffle the potential negatives and extract only 30% of them to be used as negatives
    random.shuffle(potential_negative_labels)
    
    if use_subset_of_negatives:
        print(f"Length of potential_negative_labels after shuffling and extracting {int(fraction_of_negatives_to_select*100)}% of {len(potential_negative_labels)}")
        potential_negative_labels = potential_negative_labels[:int(fraction_of_negatives_to_select*len(potential_negative_labels))]
    training_positive_samples, test_positive_samples, validation_positive_samples = train_test_val_sampler(positive_labels, train_set_pos, test_set_pos, val_set_pos)    
    if use_all_potential_negatives:
        training_negative_samples, test_negative_samples, validation_negative_samples = train_test_val_sampler(potential_negative_labels, int(0.65*len(potential_negative_labels)),
                                                                                                            int(0.25*len(potential_negative_labels)), int(0.1*len(potential_negative_labels)))    
    else:
        training_negative_samples, test_negative_samples, validation_negative_samples = train_test_val_sampler(potential_negative_labels, 3*train_set_pos, 3*test_set_pos, 3*val_set_pos)    
    
    fill_labels_and_mask(training_positive_samples, training_negative_samples ,node_gene_list, y_train, train_mask)
    fill_labels_and_mask(test_positive_samples, test_negative_samples ,node_gene_list, y_test, test_mask)
    fill_labels_and_mask(validation_positive_samples, validation_negative_samples ,node_gene_list, y_val, val_mask)  

    print(f"Number of training positives: {int(np.sum(y_train) + np.sum(y_test) + np.sum(y_val))}")
        
    return y_train, train_mask, y_test, test_mask, y_val, val_mask

def fill_labels_and_mask(positive_samples, negative_samples ,node_gene_list, y, mask):
    for gene in positive_samples:
        y[np.where(node_gene_list == gene), 0] = 1.0
        mask[np.where(node_gene_list == gene)] = 1.0 
    for gene in  negative_samples:
        mask[np.where(node_gene_list == gene)] = 1.0 
    

def feature_fold_change_pvalue_combination(log2fc, p_val, is_absolute_val):
    if p_val <= 10 ** -9:
        p_val = 10 ** -9
    if is_absolute_val:
        return np.sqrt(np.abs(log2fc * np.log10(p_val)))
    else:
        return np.sign(log2fc)*np.sqrt(np.abs(log2fc * np.log10(p_val)))
    

def find_detected_genes(dataset_type: str, proteome_file: str, transcriptome_file: str, effectome_file: str, interactome_file: str, \
                        are_missing_features_imputed :bool):
    
    df_proteomics = pd.read_csv(proteome_file)
    df_proteomics = df_proteomics.dropna(subset = ["gene_name", "fold_change_log2.SARS_CoV2@24h_vs_mock@24h", "p_value.SARS_CoV2@24h_vs_mock@24h"])
    df_proteomics["gene_name"] = df_proteomics["gene_name"].str.replace('\...','', regex=True)    
    df_proteomics["gene_name"] = df_proteomics["gene_name"].str.lower()
    
    df_transcriptomics = pd.read_csv(transcriptome_file)
    df_transcriptomics = df_transcriptomics.dropna(subset = ["gene_name", "fold_change_log2.SARS_CoV2@24h_vs_mock@24h", "p_value.SARS_CoV2@24h_vs_mock@24h"])
    df_transcriptomics["gene_name"] = df_transcriptomics["gene_name"].str.replace('\...','', regex=True)
    df_transcriptomics["gene_name"] = df_transcriptomics["gene_name"].str.lower()
    
    
    df_effectome = pd.read_csv(effectome_file)
    df_effectome = df_effectome.rename(columns={"Unnamed: 1": "viral_bait_name", "Changes vs. control overexpressed proteins": "median_log2", "Unnamed: 8": "p_value", "Unnamed: 9": "standard_dev_log2"})
    df_effectome = df_effectome.dropna(subset = ["Host protein", "median_log2", "p_value"])
    df_effectome["Host protein"] = df_effectome["Host protein"].str.replace('\...','', regex=True)
    df_effectome["Host protein"] = df_effectome["Host protein"].str.lower()
    effectome_genes_raw = df_effectome["Host protein"].unique()
    effectome_genes = []
    for gene in effectome_genes_raw:
        if ";" not in gene:
            effectome_genes.append(gene)
        else:
            effectome_genes_tmp = gene.split(";")
            for gene_tmp in effectome_genes_tmp:
                effectome_genes.append(gene_tmp)
    effectome_genes = np.unique(effectome_genes)

    
    df_interactome = pd.read_csv(interactome_file, sep = "\t")
    df_interactome["gene_names"] = df_interactome["gene_names"].str.lower()
    
    proteomics_genes = df_proteomics["gene_name"].unique()
    transcriptomics_genes = df_transcriptomics["gene_name"].unique()
    interactome_genes = df_interactome["gene_names"].unique()
    
    genes_for_input_dataset = []
    
    if dataset_type == "transcriptome-proteome-effectome-interactome":
        if not are_missing_features_imputed:
            for gene_name in effectome_genes:#proteomics_genes:
                if gene_name in transcriptomics_genes and gene_name in effectome_genes:
                    genes_for_input_dataset.append(gene_name)
        else:      
            genes_for_input_dataset = np.unique(np.concatenate((proteomics_genes, transcriptomics_genes, effectome_genes, interactome_genes), axis = None))  
                 
    if dataset_type == "transcriptome-proteome-effectome":
        if not are_missing_features_imputed:
            for gene_name in proteomics_genes:
                if gene_name in transcriptomics_genes and gene_name in effectome_genes:
                    genes_for_input_dataset.append(gene_name)
        else:      
            genes_for_input_dataset = np.unique(np.concatenate((proteomics_genes, transcriptomics_genes, effectome_genes), axis = None))
    elif dataset_type == "transcriptome":
            print(f"assigning {len(transcriptomics_genes)} transcriptomics genes")
            genes_for_input_dataset = transcriptomics_genes
    elif dataset_type == "proteome":
            print(f"assigning {len(proteomics_genes)} proteomics genes")
            genes_for_input_dataset = proteomics_genes
    elif dataset_type == "effectome":
            print(f"assigning {len(effectome_genes)} effectomics genes")
            genes_for_input_dataset = effectome_genes  
    elif dataset_type == "interactome":      
            print(f"assigning {len(interactome_genes)} virus-host interactome genes")
            genes_for_input_dataset = interactome_genes

    # set all gene names to lower case
    genes_for_input_dataset = np.array([gene_name.lower() for gene_name in genes_for_input_dataset])            
    return df_proteomics, df_transcriptomics, df_effectome, df_interactome, genes_for_input_dataset   

def extract_feature_from_effectome_row(effectome_data_frame, host_protein, viral_proteins):
    host_protein_feature_vector = []
    viral_proteins_tested_per_host_protein = effectome_data_frame[effectome_data_frame["Host protein"] == host_protein]
    for viral_protein in viral_proteins:
        fc_and_pval_per_viral_protein = viral_proteins_tested_per_host_protein[viral_proteins_tested_per_host_protein["viral_bait_name"] == viral_protein][["median_log2", "p_value"]].to_numpy()            
        host_protein_feature_vector.append(np.mean([feature_fold_change_pvalue_combination(l2fc, pval, False) for l2fc, pval in zip(fc_and_pval_per_viral_protein[:,0], fc_and_pval_per_viral_protein[:,1])]))
    return host_protein_feature_vector  

def extract_features_from_interactome_row(interactions_with_host_protein, host_protein, viral_proteins, is_used_abs_value_for_up_down_regulated: bool):
    host_protein_feature_vector = []
    viral_proteins_interacting = interactions_with_host_protein["bait_name"].unique()
    for viral_protein in viral_proteins:
        if viral_protein in viral_proteins_interacting:
            l2fc_values = interactions_with_host_protein[interactions_with_host_protein['bait_name'] == viral_protein]["fold_change_log2"].values
            p_value_values = interactions_with_host_protein[interactions_with_host_protein['bait_name'] == viral_protein]["p_value"].values
            host_protein_feature_vector.append(np.mean([feature_fold_change_pvalue_combination(l2fc, pval, False) for l2fc, pval in zip(l2fc_values,p_value_values)]))
            #host_protein_feature_vector.append(1.0)
        else:    
            host_protein_feature_vector.append(np.random.normal(0, 0.001*0.25, 1)[0] if not is_used_abs_value_for_up_down_regulated 
                                               else abs(np.random.normal(0, 0.001*0.25, 1)[0]))
    return host_protein_feature_vector        

def extract_feature_from_row(row, time_list, feature_vector, is_used_abs_value_for_up_down_regulated, fold_change_first_part :str, p_val_first_part :str):
    for time in time_list:
        l2fc = row[ fold_change_first_part + time].values
        pval = row[ p_val_first_part + time].values
        if len(l2fc) == 0:
            l2fc.append(10 ** -9)
        elif np.isnan(l2fc).any():
            l2fc[0] =  10 ** -9   
        if len(pval) == 0:
            pval.append(0.99)
        elif np.isnan(pval).any():
            pval[0] == 0.99        
        feature_vector.append(feature_fold_change_pvalue_combination( l2fc[0], pval[0], is_used_abs_value_for_up_down_regulated))
    
def compute_uninformative_features_mean_and_std(df_omic, l2fc_string = "fold_change_log2.SARS_CoV2@6h_vs_mock@6h", pval_string = "p_value.SARS_CoV2@6h_vs_mock@6h"):
    log2fcs = df_omic[l2fc_string].values           
    pvalues = df_omic[pval_string].values
    
    vectorized_feature_fold_change_pvalue_combination = np.vectorize(feature_fold_change_pvalue_combination)
    features = vectorized_feature_fold_change_pvalue_combination(log2fcs, pvalues, is_absolute_val=False)
    return np.nanmean(features), np.nanstd(features)

def append_uninformative_features(feature_vector, mu, sigma, is_used_abs_value_for_up_down_regulated: bool, is_append_zero_imputed_features: bool, sigma_fraction_to_consider, number_of_considered_timestamps = 3):
    for i in range(number_of_considered_timestamps):
            if is_append_zero_imputed_features:
                feature_vector.append(0.0)
            elif is_used_abs_value_for_up_down_regulated:
                feature_vector.append(abs(np.random.normal(mu, sigma*sigma_fraction_to_consider, 1)[0])) 
            else:                 
                feature_vector.append(np.random.normal(mu, sigma*sigma_fraction_to_consider, 1)[0])


def assign_features_to_genes_sars_cov_2(df_transcriptomics, df_proteomics, df_effectome, df_interactome, connected_common_genes,
                             timestamp_list_transcriptomics, timestamp_list_proteomics, viral_proteins,
                             is_used_abs_value_for_up_down_regulated, is_append_zero_imputed_features,
                             ):

    # Compute uninformative features mean and standard deviation for each dataset
    mu_transcriptomics, sigma_transcriptomics = compute_uninformative_features_mean_and_std(df_transcriptomics)
    mu_proteomics, sigma_proteomics = compute_uninformative_features_mean_and_std(df_proteomics)
    mu_effectome, sigma_effectome = compute_uninformative_features_mean_and_std(df_effectome, l2fc_string="median_log2", pval_string="p_value")
    
    # Initialize variables
    interactome_counter = 0
    feature_dict = {}
    genes_with_imputed_feature_values = []

    # Iterate over connected common genes
    for gene in tqdm(connected_common_genes, desc="Assigning features to connected genes"):
        feature_vector = []
        
        # Transcriptomics
        row = df_transcriptomics[df_transcriptomics["gene_name"] == gene]
        if len(row) == 0:
            genes_with_imputed_feature_values.append(gene)
            append_uninformative_features(feature_vector, mu_transcriptomics, sigma_transcriptomics, is_used_abs_value_for_up_down_regulated, is_append_zero_imputed_features, sigma_fraction_to_consider=0.25)
        else:
            extract_feature_from_row(row, timestamp_list_transcriptomics, feature_vector, is_used_abs_value_for_up_down_regulated, "fold_change_log2", "p_value")
        
        # Proteomics
        row = df_proteomics[df_proteomics["gene_name"] == gene]
        if len(row) == 0:
            if len(genes_with_imputed_feature_values) == 0 or genes_with_imputed_feature_values[-1] != gene:
                genes_with_imputed_feature_values.append(gene)
            append_uninformative_features(feature_vector, mu_proteomics, sigma_proteomics, is_used_abs_value_for_up_down_regulated, is_append_zero_imputed_features, sigma_fraction_to_consider=0.25)
        else:
            extract_feature_from_row(row, timestamp_list_proteomics, feature_vector, is_used_abs_value_for_up_down_regulated, "fold_change_log2", "p_value")
        
        # Effectome
        row = df_effectome[df_effectome["Host protein"] == gene]
        if len(row) == 0:
            if len(genes_with_imputed_feature_values) == 0 or genes_with_imputed_feature_values[-1] != gene:
                genes_with_imputed_feature_values.append(gene)
            [append_uninformative_features(feature_vector, mu_effectome, sigma_effectome, is_used_abs_value_for_up_down_regulated, is_append_zero_imputed_features, sigma_fraction_to_consider=0.25, number_of_considered_timestamps=1) for _ in range(len(viral_proteins))]
        else:
            effectome_features = extract_feature_from_effectome_row(df_effectome, gene, viral_proteins)
            feature_vector = np.concatenate((feature_vector, effectome_features), axis=0).tolist()
        
        # Interactome
        row = df_interactome[df_interactome["gene_names"] == gene]
        if len(row) == 0:
            interactome_counter += 1
            if len(genes_with_imputed_feature_values) == 0 or genes_with_imputed_feature_values[-1] != gene:
                genes_with_imputed_feature_values.append(gene)
            [append_uninformative_features(feature_vector, 0.0, 0.001, is_used_abs_value_for_up_down_regulated, is_append_zero_imputed_features, sigma_fraction_to_consider=0.25, number_of_considered_timestamps=1) for _ in range(len(viral_proteins))]
        else:
            assert len(df_interactome[df_interactome["gene_names"] == gene].values[0][1:]) == 24
            interactome_features = df_interactome[df_interactome["gene_names"] == gene].values[0][1:]
            feature_vector = np.concatenate((feature_vector, interactome_features), axis=0).tolist()

        # Update feature dictionary
        feature_dict.update({gene: feature_vector})

    return feature_dict, genes_with_imputed_feature_values, interactome_counter

def assign_features_to_genes_mpxv(df_transcriptomics, df_proteomics, connected_common_genes, timestamp_list):
    
    feature_dict = {}
    genes_with_imputed_feature_values = []

    for gene in tqdm(connected_common_genes, desc="Assigning features to connected genes"):
        feature_vector = []
        
        # Transcriptomics
        row = df_transcriptomics[df_transcriptomics["gene_name"].str.lower() == gene]
        if len(row) == 0:
            genes_with_imputed_feature_values.append(gene)
            append_uninformative_features(feature_vector, 0.0, 0.001, is_used_abs_value_for_up_down_regulated=False, is_append_zero_imputed_features=True, sigma_fraction_to_consider=0.25, number_of_considered_timestamps=3)
        else:
            extract_feature_from_row(row, timestamp_list, feature_vector, is_used_abs_value_for_up_down_regulated=False, fold_change_first_part='fold_change_log2', p_val_first_part='p_value')
        # Proteomics
        row = df_proteomics[df_proteomics["gene_names"].str.lower() == gene]
        if len(row) == 0:
            if len(genes_with_imputed_feature_values) == 0 or genes_with_imputed_feature_values[-1] != gene:
                genes_with_imputed_feature_values.append(gene)
            append_uninformative_features(feature_vector, 0.0, 0.001, is_used_abs_value_for_up_down_regulated=False, is_append_zero_imputed_features=True, sigma_fraction_to_consider=0.25, number_of_considered_timestamps=3)
        else:
            extract_feature_from_row(row, timestamp_list, feature_vector, is_used_abs_value_for_up_down_regulated=False, fold_change_first_part='fold_change_log2', p_val_first_part='p_value')

        feature_dict.update({gene: feature_vector})
    
    return feature_dict, genes_with_imputed_feature_values


def proteomics_trascriptomics_effectome_interactome_features_prep(timestamp_list_transcriptomics, timestamp_list_proteomics, is_used_abs_value_for_up_down_regulated :bool, \
                                                                  is_append_zero_imputed_features: bool, return_detected_genes, df_ppi, are_missing_features_imputed: bool):
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        df_proteomics, df_transcriptomics, df_effectome, df_interactome, common_genes = return_detected_genes()
        viral_proteins = df_effectome["viral_bait_name"].unique()
        global_variables.viral_proteins = viral_proteins
    elif global_variables.fine_tuning_virus == "MPXV":
        df_proteomics, df_transcriptomics = global_variables.df_proteome_isoforms_filtered_mpxv, global_variables.df_transcriptome_mpxv
        common_genes = global_variables.common_genes_mpxv

    connected_common_genes = []
    connected_genes = np.unique(np.concatenate((df_ppi["gene_name_1"].values, df_ppi["gene_name_2"].values), axis=0))
    global_variables.connected_genes = connected_genes
    for gene in tqdm(common_genes, desc="Selecting connected genes among genes detected in the data"):
        if gene in connected_genes and not are_missing_features_imputed:
            connected_common_genes.append(gene)
        elif are_missing_features_imputed:
            connected_common_genes = connected_genes
    
    feature_dict = {}
    genes_with_imputed_feature_values = []
    if not debug["feature_dict_prep"]:
        print("Assigning precomputed features")
        feature_dict = np.load(artivir_data_path + 'feature_dict_tmp.npy',allow_pickle='TRUE').item()
        genes_with_imputed_feature_values = np.load(artivir_data_path + 'genes_with_imputed_feature_values_tmp.npy',allow_pickle='TRUE')
    else:
        if global_variables.fine_tuning_virus == "SARS-CoV-2":    
            feature_dict, genes_with_imputed_feature_values, interactome_counter = assign_features_to_genes_sars_cov_2(df_transcriptomics, 
                                                                                                                       df_proteomics, df_effectome, 
                                                                                                                       df_interactome, connected_common_genes, 
                                                                                                                       timestamp_list_transcriptomics, timestamp_list_proteomics, 
                                                                                                                       viral_proteins, is_used_abs_value_for_up_down_regulated, 
                                                                                                                       is_append_zero_imputed_features)
            print("genes not in interactome percentage: ", 100 * interactome_counter/len(feature_dict.keys()), "%")
        elif global_variables.fine_tuning_virus == "MPXV":
            feature_dict, genes_with_imputed_feature_values = assign_features_to_genes_mpxv(df_transcriptomics, df_proteomics, connected_common_genes, timestamp_list_transcriptomics)  

        np.save(artivir_data_path + 'genes_with_imputed_feature_values_tmp.npy', genes_with_imputed_feature_values)
        np.save(artivir_data_path + 'feature_dict_tmp.npy', feature_dict)
             
    return feature_dict, genes_with_imputed_feature_values         
    

def multiomics_ppi_prep(return_detected_genes, dataset_type: str, input_graph_type: str, are_missing_features_imputed: bool, binary_compound_score_values: list[int]):

    data_path = artivir_data_path    
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        _, _, _, _, detected_genes  =  return_detected_genes()
    else:
        detected_genes = global_variables.common_genes_mpxv
    
    print(f"The total number of detected genes is: {len(detected_genes)}")
    if input_graph_type == "STRING":  
        print("Opening STRINGdb")     
        weights = binary_compound_score_values
        df_ppi = aggregate_scores(global_variables.storage_folder + "9606.protein.links.full.v12.0.txt.gz", weights=weights) #artivir_data_path + "df_string_ppi_complete.zip", weights=weights) #
    elif input_graph_type == "mulstiscale_interactome":
        print("Opening multiscale_interactome PPI")    
        df_ppi = pd.read_csv(data_path + "df_multiscale_interactome.zip")

    if debug["simple_debug"]:
        number_of_connections = 500000
        print(f"Debug mode: selecting only {number_of_connections} PPIs")
        df_ppi = df_ppi.head(number_of_connections)

    # lowercase all gene names
    df_ppi["gene_name_1"] = df_ppi["gene_name_1"].str.lower()
    df_ppi["gene_name_2"] = df_ppi["gene_name_2"].str.lower()

    if not are_missing_features_imputed:
        print("Selecting only genes detected by " + dataset_type + " and dropping the rest")
        detected_genes_set = set(detected_genes)
        mask_gene_1 = df_ppi["gene_name_1"].isin(detected_genes_set)
        mask_gene_2 = df_ppi["gene_name_2"].isin(detected_genes_set)
        df_ppi = df_ppi[mask_gene_1 & mask_gene_2]
        df_ppi.reset_index(drop = True, inplace = True)
    
    return df_ppi
          

def add_gene_names_to_ppi(data_path: str = artivir_data_path):
    if os.path.isfile(data_path + "/df_string_ppi_with_gene_names.zip"):
        return
    gene_to_string_names_map_path = data_path + "/df_string_to_gen_names_map.zip"    
    if not os.path.isfile(gene_to_string_names_map_path):
        print("df_string_to_gen_names_map.zip was not computed")
        return
        
    df_string_to_gen_names_map = pd.read_csv(gene_to_string_names_map_path)
    df_string_ppi = pd.read_csv(data_path + "/df_string_ppi.zip")
    gene_names_1 = []
    gene_names_2 = []

    for _, row in tqdm(df_string_ppi.iterrows(), total=df_string_ppi.shape[0]):
        names_1 = df_string_to_gen_names_map[df_string_to_gen_names_map["string_ids"] == row["protein1"]]["gene_names"]
        names_2 = df_string_to_gen_names_map[df_string_to_gen_names_map["string_ids"] == row["protein2"]]["gene_names"]
        if len(names_1) > 0:
            gene_names_1.append(names_1.values[0])
        else:
            gene_names_1.append('nan')
        if len(names_2) > 0:    
            gene_names_2.append(names_2.values[0])
        else:
            gene_names_2.append('nan')
            
    df_string_ppi["gene_name_1"] = pd.Series(gene_names_1, index = df_string_ppi.index)     
    df_string_ppi["gene_name_2"] = pd.Series(gene_names_2, index = df_string_ppi.index)  
    df_string_ppi.to_csv(data_path + "/df_string_ppi_with_gene_names.zip", index=False, compression = dict(method='zip',archive_name='df_string_ppi_with_gene_names.csv'))
    

def map_ensp_to_gene_name(uniprot_file_path :str, data_path :str = artivir_data_path):
    if os.path.isfile(data_path + "df_string_to_gen_names_map.zip"):
        return
    if not os.path.isfile(data_path + "df_string_ppi.zip"):
        print("df_string_ppi.zip file not available")
        return
    
    uniprot_dat_file_content = [i.strip().split() for i in open(uniprot_file_path).readlines()]
    string_and_gene_names = []
    for item in uniprot_dat_file_content:
        if (item[1] == "Gene_Name") or (item[1] == "STRING"):
            string_and_gene_names.append(item)    
    
    string_names_list = []
    gene_names_list = []
    for elem in tqdm(string_and_gene_names, desc="look for STRING protein ids corresponding to gene names in dataset"):
        if elem[1] == "STRING":
            for names in string_and_gene_names: # check if the uniprot name is the same
                if names[0] == elem[0] and names[1] == "Gene_Name":
                    string_names_list.append(elem[2])
                    gene_names_list.append(names[2])
                    break     
            
    string_to_gen_names_map = {"string_ids" : string_names_list, "gene_names" : gene_names_list}        
    df_string_to_gen_names_map = pd.DataFrame(string_to_gen_names_map)
    df_string_to_gen_names_map.to_csv(data_path + 'df_string_to_gen_names_map.zip', index=False, compression = dict(method='zip',archive_name='df_string_to_gen_names_map.csv'))
            

def import_and_filter_string_ppi(path_to_string :str):
    data_path = artivir_data_path
    if os.path.isfile(data_path + "/GhostFreePro/data_preprocessing_pipeline/artivir_data/df_string_ppi.zip"):
        return
    
    df_string_ppi = pd.read_csv(path_to_string , sep = " ")
    # print("removing the experimental edjes")
    # counter = 0
    # index_list = []
    # print(len(df_string_ppi))
    # for index, row in df_string_ppi.iterrows():
    #     if row["experimental"] < 10:
    #         counter += 1
    #         index_list.append(index)
    #         if counter % 10000 == 0:    
    #             print(index) 
                 
    # df_string_ppi.drop(index_list, axis = 0, inplace = True) 
    # df_string_ppi.reset_index(drop = True, inplace = True) 
    print("saving the filtered ppi")
    df_string_ppi.to_csv(data_path + 'df_string_ppi.zip', index=False, compression = dict(method='zip',archive_name='df_string_ppi.csv'))

def data_preprocessing(input_data_path):
    data_path = artivir_data_path
    if os.path.isfile(data_path + "transcriptome.zip") \
        and os.path.isfile(data_path + "proteome_cell_lines.zip") \
        and os.path.isfile(data_path + "host_factors_to_remove.zip") \
        and os.path.isfile(data_path + "effectome.zip") \
        and os.path.isfile(data_path + "viral_host_interactome.zip"):
        return
    
    print("Input file reading")
    data = pd.ExcelFile(input_data_path)

    dfs =  {sheet_name: data.parse(sheet_name) for sheet_name in data.sheet_names}

    transcriptome = dfs["Tx"]
    proteome_cell_lines = dfs["FP"]
    effectome = dfs["EFF"][dfs["EFF"]["Viral protein"] == "SARS-CoV-2"]
    viral_host_interactome = dfs["PPIs"]
    
    
    transcriptome.to_csv(artivir_data_path + 'transcriptome.zip', index=False, compression = dict(method='zip',archive_name='transcriptome.csv'))
    proteome_cell_lines.to_csv(artivir_data_path + 'proteome_cell_lines.zip', index=False, compression = dict(method='zip',archive_name='proteome_cell_lines.csv'))
    effectome.to_csv(artivir_data_path + 'effectome.zip', index=False, compression = dict(method='zip',archive_name='effectome.csv'))
    viral_host_interactome.to_csv(artivir_data_path + 'viral_host_interactome.zip', index=False, compression = dict(method='zip',archive_name='viral_host_interactome.csv'))

def parse_args(artivir_data_dir):
    parser = argparse.ArgumentParser(description='Data preprocessing for transcriptome and proteome from cell lines')

    parser.add_argument('--input_data_path', help='Artivir file path',
                    default=artivir_data_dir + "ARTIvir_CoV_minimal_combined_dset.xlsx",
                    type=str
                    )
    parser.add_argument('--potential_host_factors_from_publications', help='Host factors file data path',
                    default=artivir_data_dir + "host_factors_from_publications.xlsx",
                    type=str
                    )    
    parser.add_argument('--path_to_string_db', help='STRING ppi network file path',
                    default= artivir_data_dir + "9606.protein.links.full.v11.5.txt",#"9606.protein.physical.links.detailed.v11.5.txt",
                    type=str
                    )    
    parser.add_argument('--uniprot_file_path', help = 'Uniprot download file path',
                    default = artivir_data_dir + "HUMAN_9606_idmapping.dat",
                    type=str
                    )   
    parser.add_argument('--strong_host_factors', help = 'Host factors to consider as positive labels',
                    default = artivir_data_dir + "strong_host_factors_list_from_publications.npy",
                    type=str
                    )  
    parser.add_argument('--host_factors_shared_from_stukalov_and_others', help = 'Host factors shared between stukalov and other indipendent studies to consider as positive labels',
                    default = artivir_data_dir + "host_factors_stukalov_plus_two_more_studies.txt",
                    type=str
                    )
    parser.add_argument( "dataset_type", type = str, choices=["transcriptome", "proteome", "effectome", "transcriptome-proteome-effectome"],
                    help="Choose which data you want to use to create the input dataset for EMOGI")
    
    parser.add_argument('--is_trivial_features', action='store_true', default = False, help = "Use trivial features to test the negative vs. positives classifier")

    parser.add_argument('--hdf5_file_name', type = str, help = "Specify hdf5 file name")    
    
    args = parser.parse_args()
    
    if args.hdf5_file_name is None:
        args.hdf5_file_name = args.dataset_type
    return args 

def process_data(is_used_abs_value_for_up_down_regulated, is_append_zero_imputed_features, return_detected_genes, df_ppi, 
                 are_missing_features_imputed, omic_data_type, debug, is_dataset_emogi_compatible, interaction_score_threshold, 
                 strong_host_factors, host_factors_shared_from_stukalov_and_others, 
                 validation_set_genes_path, are_stukalov_shared_genes_used, disease_genes_related_to_viral_infections):
    potential_host_factors_from_publications = global_variables.host_factors_from_publications
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        time_stamp_list = [".SARS_CoV2@6h_vs_mock@6h", ".SARS_CoV2@12h_vs_mock@12h", ".SARS_CoV2@24h_vs_mock@24h"]
    elif global_variables.fine_tuning_virus == "MPXV":
        time_stamp_list = [".6h",".12h",".24h"] 
    feature_dict, _ = proteomics_trascriptomics_effectome_interactome_features_prep(time_stamp_list, time_stamp_list, 
                                                                                                                is_used_abs_value_for_up_down_regulated,
                                                                                                                is_append_zero_imputed_features,
                                                                                                                return_detected_genes=return_detected_genes,
                                                                                                                df_ppi=df_ppi,
                                                                                                                are_missing_features_imputed=are_missing_features_imputed)
    network, features, node_names, feat_names, _, randomized_gene_list = create_adjacency_matrix_and_feature_matrix(df_ppi, feature_dict, 
                                                                                                                omic_data_type,
                                                                                                                is_debug = debug["simple_debug"],
                                                                                                                is_dataset_emogi_compatible = is_dataset_emogi_compatible,
                                                                                                                interaction_score_threshold = interaction_score_threshold)
        
    y_train, train_mask, y_test, test_mask, y_val, val_mask = train_test_val_split(randomized_gene_list, potential_host_factors_from_publications,
                                                                                   strong_host_factors, host_factors_shared_from_stukalov_and_others,
                                                                                   validation_set_genes_path, 
                                                                                   are_stukalov_shared_genes_used,
                                                                                   disease_genes_related_to_viral_infections,
                                                                                   use_all_potential_negatives = True)   
    
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names

def create_torch_tensor_data_obj(network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names, omic_specific_type : str):
    data = Data()
    data.edge_index = torch.Tensor([network[0], network[1]]).long()
    data.edge_weight = torch.Tensor(network[2])
    print("Use omic type " + omic_specific_type)
    data.x = torch.tensor(features).float()
    num_node_features = 0
    if "transcriptome" in omic_specific_type:    
        num_node_features += torch.tensor(np.shape(features[:,0:3])[1]).long().item()
    if "proteome" in omic_specific_type:
        num_node_features += torch.tensor(np.shape(features[:,3:6])[1]).long().item()
    if "effectome" in omic_specific_type:
        num_node_features += torch.tensor(np.shape(features[:,6:30])[1]).long().item()
    if "interactome" in omic_specific_type:
        num_node_features += torch.tensor(np.shape(features[:,30:54])[1]).long().item()
    data.num_node_features = num_node_features
    
    data.y = torch.tensor(y_val.transpose()[0] + y_test.transpose()[0] + y_train.transpose()[0]).long()
    data.train_mask = torch.tensor(train_mask).bool()
    data.test_mask = torch.tensor(test_mask).bool()
    data.val_mask = torch.tensor(val_mask).bool()
    data.node_names = node_names
    data.feat_names = feat_names
    
    return data 

def run_artivir_data_preparation(artivir_data_dir: str, strong_host_factors: str,
                                 omic_data_type: str, host_factors_shared_from_stukalov_and_others: str,
                                 is_trivial_features: bool, hdf5_file_name: str, is_used_abs_value_for_up_down_regulated: bool,
                                 are_missing_features_imputed: bool, input_graph_type: str, are_stukalov_shared_genes_used: bool,
                                 is_append_zero_imputed_features: bool, interaction_score_threshold: float, validation_set_genes_path: str,
                                 binary_compound_score_values: list[int], disease_genes_related_to_viral_infections: str):    
    
    hdf5_file_name = hdf5_file_name + "_abs_feature_values_" + str(is_used_abs_value_for_up_down_regulated) \
                                                     + "_are_missing_values_imputed_" + str(are_missing_features_imputed) + "_" \
                                                     + input_graph_type                                           
    
    if is_append_zero_imputed_features:
        hdf5_file_name = hdf5_file_name + '_' + 'zero_imputed'

    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        return_detected_genes = lambda: find_detected_genes(omic_data_type, artivir_data_dir + "proteome_cell_lines.zip",artivir_data_dir + "transcriptome.zip", \
                                                        artivir_data_dir + "effectome.zip",artivir_data_dir + "virus_host_interactome.csv", \
                                                        are_missing_features_imputed)
    else:
        global_variables.load_mpxv_data()
        return_detected_genes = global_variables.common_genes_mpxv
    # debug save of precomputed omic specific PPI
    if not debug["multiomics_ppi_prep"]:
        print("Reading df_ppi_tmp")
        df_ppi = pd.read_csv(artivir_data_path + 'df_ppi_tmp.zip')
    else:    
        print("Re-reading PPI network")
        df_ppi = multiomics_ppi_prep(return_detected_genes, omic_data_type, input_graph_type, are_missing_features_imputed, binary_compound_score_values)

    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = process_data(is_used_abs_value_for_up_down_regulated, is_append_zero_imputed_features, return_detected_genes, df_ppi,
                        are_missing_features_imputed, omic_data_type, debug, is_dataset_emogi_compatible, interaction_score_threshold,
                         strong_host_factors, host_factors_shared_from_stukalov_and_others,
                        validation_set_genes_path, are_stukalov_shared_genes_used, disease_genes_related_to_viral_infections)      
    
    data = create_torch_tensor_data_obj(network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names, omic_data_type)
    return data
    
def process_disease_genes(data, antiviral_drug_targets_to_remove):
    if global_variables.is_pretraining_on_all_diseases:
        inference_score_dict = global_variables.load_pretraining_genes_from_all_diseases()
        direct_evidence_threshold = 1
    else:
        direct_evidence_threshold = 0
        inference_score_dict = global_variables.load_pretraining_genes_from_viral_diseases()

    mean_inference_score = np.nanmean(inference_score_dict["inference_score"])
    std_inference_score = np.nanstd(inference_score_dict["inference_score"])
    inference_score_threshold = mean_inference_score + 2*std_inference_score
        
    positive_disease_genes = []
    negative_disease_genes = []
    direct_evidence_genes = []
    high_score_genes = []

    # retrieve the negative disease genes from the inference score dict
    ppi_genes = data.node_names[:,0].astype(str)
    print(f"Train mask size before removing the genes that should not be considered as negatives in the fine tuning: {data.train_mask.sum().item()}")
    for gene, inference_score, direct_evidence in zip(inference_score_dict["gene"], \
                                                      inference_score_dict["inference_score"], \
                                                        inference_score_dict["direct_evidence"]):
        # find genes that should not be considered as negatives in the fine tuning
        if inference_score > mean_inference_score + std_inference_score or direct_evidence > 0:
            if gene.lower() in ppi_genes:
                gene_index = np.where(ppi_genes == gene.lower())[0][0]
                #check that the gene is not an host factor used in fine tuning
                if data.y[gene_index] == 0:
                    # remove the genes that should not be considered as negatives in the fine tuning
                    data.train_mask[gene_index] = False
                    data.val_mask[gene_index] = False
                    data.test_mask[gene_index] = False
        if inference_score > (mean_inference_score + (3/2)*std_inference_score) or direct_evidence > direct_evidence_threshold:
            if gene.lower() not in antiviral_drug_targets_to_remove:
                positive_disease_genes.append(gene.lower()) 
                if direct_evidence > direct_evidence_threshold:
                    direct_evidence_genes.append(gene.lower())
                elif inference_score > inference_score_threshold:
                    high_score_genes.append(gene.lower())
            else:
                print(f"Gene {gene} is an antiviral drug target and will be removed from the positive disease genes")
        elif inference_score <= (mean_inference_score + std_inference_score) and direct_evidence <= direct_evidence_threshold:
            negative_disease_genes.append(gene.lower())
    
    print(f"Train mask size after removing the genes {data.train_mask.sum().item()}")

    inference_score_genes = [gene.lower() for gene in inference_score_dict["gene"]]

    # complete the negative disease genes with the ones present in the ppi network but not in the inference score dict from CTD
    for gene in ppi_genes:
        if gene not in inference_score_genes:
            negative_disease_genes.append(gene.lower())

    # filter from the negative disease genes the ones that are listed as host factors from publications 
    host_factors_to_remove = global_variables.host_factors_from_publications
    negative_disease_genes = np.array([gene for gene in negative_disease_genes if gene not in host_factors_to_remove])


    #select for training positives with not too high degree and negatives with high degree and exclude all the positive genes from the negatives
    name_to_index = {gene: index for index, gene in enumerate(ppi_genes)}
    index_to_name = {index: gene for index, gene in enumerate(ppi_genes)}
    # save name_to_index map in global vars for later use
    global_variables.name_to_index = name_to_index
    positive_indices = [name_to_index[name] for name in positive_disease_genes if name in ppi_genes]
    negative_indices = [name_to_index[name] for name in negative_disease_genes if name in ppi_genes]
    direct_evidence_indices = [name_to_index[name] for name in direct_evidence_genes if name in ppi_genes]
    high_score_indices = [name_to_index[name] for name in high_score_genes if name in ppi_genes]

    positive_indices = list(map(int, positive_indices))
    negative_indices = list(map(int, negative_indices))
    direct_evidence_indices = list(map(int, direct_evidence_indices))
    high_score_indices = list(map(int, high_score_indices))
    
    degree = pyg_utils.degree(data.edge_index[0], num_nodes=data.num_nodes)
    # save the degree dsitribution in global vrariables for later use
    # create a dictionary with the gene names as keys and the degrees as values
    global_variables.degree_dict = {gene: degree[index].item() for index, gene in enumerate(ppi_genes)}
    mean_degree = degree.mean()
    # select positives with not too high degree and include all the direct evidence genes
    potential_positives_genes = []
    positive_indices = [index for index in positive_indices if degree[index] < mean_degree + int(mean_degree/2)]
    positive_indices = np.unique(np.concatenate((positive_indices, direct_evidence_indices, high_score_indices)))
    positive_selected_genes = ppi_genes[positive_indices]
    # consider the also the exluded high degree genes as potential positives together with the low degree disease related genes
    for gene in positive_disease_genes: 
        potential_positives_genes.append(gene)
    # select not disconnected negatives 
    negative_disease_genes = [ppi_genes[index] for index in negative_indices if degree[index] > 1]
    # select negatives with not too low degree
    negative_disease_genes = [gene for gene in negative_disease_genes if degree[name_to_index[gene]] > 5] #mean_degree - int(2*mean_degree/3)]
    print(f"Number of negative selected genes: {len(negative_disease_genes)} with mean degree: {int(degree[[name_to_index[name] for name in negative_disease_genes if name in ppi_genes]].mean().item())}")

    for gene in ppi_genes:
        if gene not in negative_disease_genes:
            potential_positives_genes.append(gene)

    potential_positives_genes = np.unique(np.array(potential_positives_genes))

    # save the positive selected genes to file
    with open(global_variables.storage_folder + "positive_selected_genes.txt", "w") as file:
        for gene in positive_selected_genes:
            file.write(gene + "\n")

    # open functionally validated host factors list and remove it from the positives
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        functionally_validated_host_factors = np.load(artivir_data_path + "strong_functionally_validated_host_factors.npy")
        functionally_validated_host_factors = np.array([gene.lower() for gene in functionally_validated_host_factors])
    else:
        functionally_validated_host_factors = global_variables.host_factors_mpxv

    # remove the functionally validated host factors from the positive selected genes
    positive_selected_genes = np.array([gene for gene in positive_selected_genes if gene not in functionally_validated_host_factors])
    print(f"Number of positive selected genes: {len(positive_selected_genes)} with mean degree: {int(degree[positive_indices].mean())}")

    return positive_selected_genes, potential_positives_genes, ppi_genes, degree

def process_drug_target_genes(drug_target_genes_file_path: str, ppi_genes, degree, antiviral_drug_targets_to_remove, positive_selected_genes):
    drug_target_genes = np.loadtxt(drug_target_genes_file_path, dtype=str)
    # check that all the genes in the drug target genes file are present in the ppi network and not in the antiviral drug targets used for validation
    drug_target_genes = np.array([gene.lower() for gene in drug_target_genes if gene.lower() in ppi_genes])
    drug_target_genes_filtered = []
    for gene in drug_target_genes:
        if gene not in antiviral_drug_targets_to_remove:
            drug_target_genes_filtered.append(gene)
        else:
            print(f"Gene {gene} is an antiviral drug target and will be removed from the drug target genes")
    # check that the drug target genes are not too high degree
    name_to_index = global_variables.name_to_index
    drug_target_indices = [name_to_index[name] for name in drug_target_genes_filtered if name in ppi_genes]
    mean_degree = degree.mean()
    drug_target_indices = [index for index in drug_target_indices if degree[index] < mean_degree + int(mean_degree/5)]
    drug_target_genes_filtered = [ppi_genes[index] for index in drug_target_indices]
    potential_drug_target_genes = []
    # add the drug target genes to the potential drug target genes not to be selected as negatives
    for gene in drug_target_genes:
        potential_drug_target_genes.append(gene) 
    # add the positive selected genes to the potential drug target genes
    for gene in positive_selected_genes:
        potential_drug_target_genes.append(gene)
    potential_drug_target_genes = np.unique(np.array(potential_drug_target_genes))

    # open functionally validated host factors list and remove it from the positives
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        functionally_validated_host_factors = np.load(artivir_data_path + "strong_functionally_validated_host_factors.npy")
        functionally_validated_host_factors = np.array([gene.lower() for gene in functionally_validated_host_factors])
    elif global_variables.fine_tuning_virus == "MPXV":
        functionally_validated_host_factors = global_variables.host_factors_mpxv

    # remove the functionally validated host factors from the drug target genes
    #print(f"Number of drug target genes before removing the functionally validated host factors: {len(drug_target_genes_filtered)}")
    #drug_target_genes_filtered = [gene for gene in drug_target_genes_filtered if gene not in functionally_validated_host_factors]

    print(f"Number of drug target genes: {len(drug_target_genes_filtered)} with mean degree: {int(degree[drug_target_indices].mean())}")

    drug_target_genes_filtered = np.array(drug_target_genes_filtered)
    print(f"Number of potential drug target genes not to be selected as negatives: {len(potential_drug_target_genes)}")
    return drug_target_genes_filtered, potential_drug_target_genes

def run_pretraining_data_preparation(pe_encoding_file: np.ndarray, genes_ppi_pe: str, \
                                    go_sem_sim_matrix_reduced: np.ndarray, genes_gosemsim: str, data_ft: Data, \
                                    protein_embeddings_file_path: str):
        
    # load ppe encoding file
    pe_ppr = pe_encoding_file
    genes_ppi_pe = np.array([gene.lower() for gene in genes_ppi_pe])
    # load go semantic matrix file 
    go_sem_matrix = go_sem_sim_matrix_reduced
    genes_gosemsim = np.array([gene.lower() for gene in genes_gosemsim])
    # load protein embeddings file
    protein_embeddings = pickle.load(open(protein_embeddings_file_path, "rb"))
    protein_embeddings_genes = protein_embeddings["gene"].values
    protein_embeddings_genes = np.array([gene.lower() for gene in protein_embeddings_genes])
    protein_embeddings_values = protein_embeddings['facebook/esm2_t12_35M_UR50D_embedding'].values
    protein_embeddings_tensor = [torch.tensor(elem) for elem in protein_embeddings_values]
    protein_embeddings_tensor = torch.stack(protein_embeddings_tensor)

    # retrive the positive and negative disease genes from the inference score dict using a lower and upper bound on the inference score 
    antiviral_drug_targets_to_remove = np.loadtxt(global_variables.storage_folder + "anti_viral_drug_targets.txt", dtype=str)
    positive_disease_genes, potential_positives_genes, ppi_genes, degree = process_disease_genes(data_ft, antiviral_drug_targets_to_remove)
    
    # retrieve the drug target genes from the drug target genes file
    drug_target_genes, potential_drug_targets = process_drug_target_genes(global_variables.storage_folder + 'drug_bank/drug_targets.txt', ppi_genes, degree, antiviral_drug_targets_to_remove, positive_disease_genes)

    # concatenate pe_ppr and go_sem_matrix to create the feature matrix 
    features = torch.zeros((ppi_genes.shape[0], pe_ppr.shape[1] + go_sem_matrix.shape[1] + protein_embeddings_tensor.shape[1]))
    # create a lookup table for the genes in the gosemsim matrix and the genes in the ppi positional encoding matrix and genes in the protein embedding tensor using the genes in the ppi as reference
    gene_indices_dict = {gene_name: index for index, gene_name in enumerate(ppi_genes)}
    gosemsim_indices_dict = {gene_name: index for index, gene_name in enumerate(genes_gosemsim)}
    ppi_pe_indices_dict = {gene_name: index for index, gene_name in enumerate(genes_ppi_pe)}
    protein_embeddings_indices_dict = {gene_name: index for index, gene_name in enumerate(protein_embeddings_genes)}
    
    print("filling the feature matrix with the ppi positional encoding values")
    for gene in genes_ppi_pe:
        if gene in ppi_genes:
            features[gene_indices_dict[gene],0:pe_ppr.shape[1]] = pe_ppr[ppi_pe_indices_dict[gene],:]
    print("filling the feature matrix with the go semantic matrix values")
    for gene in genes_gosemsim:
        if gene in ppi_genes:
            features[gene_indices_dict[gene],pe_ppr.shape[1]:pe_ppr.shape[1] + go_sem_matrix.shape[1]] = go_sem_matrix[gosemsim_indices_dict[gene],:] 
    print("filling the feature matrix with the protein embeddings values")
    for gene in protein_embeddings_genes:
        if gene in ppi_genes:
            features[gene_indices_dict[gene], pe_ppr.shape[1] + go_sem_matrix.shape[1]:] = protein_embeddings_tensor[protein_embeddings_indices_dict[gene],:]        

    data = Data()
    data.edge_index = data_ft.edge_index
    data.edge_weight = data_ft.edge_weight
    data.x = features
    data.x_ft = data_ft.x
    data.num_node_features = features.shape[1]
    data.node_names = ppi_genes

    # set fine-tuning labels and masks when training with preprocessing data
    data.train_mask_ft = data_ft.train_mask
    data.test_mask_ft = data_ft.test_mask
    data.val_mask_ft = data_ft.val_mask
    data.y_ft = data_ft.y

    # create the splits in case of diesease genes
    y_train, train_mask, y_test, test_mask, y_val, val_mask = train_test_val_split(node_gene_list=ppi_genes,
                                                                                potential_host_factors_from_publications=potential_positives_genes, 
                                                                                strong_host_factors=positive_disease_genes,
                                                                                host_factors_shared_from_stukalov_and_others="",
                                                                                validation_set_genes_path="", 
                                                                                are_stukalov_shared_genes_used=False,
                                                                                use_all_potential_negatives=True,
                                                                                use_subset_of_negatives=False,
                                                                                is_virus_specific_model=False,
                                                                                is_pretraining_split=True)
    
    # set pre-training labels and masks when training with pre-training data 
    data.train_mask = torch.tensor(train_mask).bool()
    data.test_mask = torch.tensor(test_mask).bool()
    data.val_mask = torch.tensor(val_mask).bool()
    data.y = torch.tensor(y_val.transpose()[0] + y_test.transpose()[0] + y_train.transpose()[0]).long()
    print("Number of nodes: ", data.num_nodes)
    print("Number of edges: ", data.num_edges)
    print("Number of features: ", data.num_node_features)
    print("Number of positive labels: ", data.y.sum().item())
    print("Number of negative labels: ", data.train_mask.sum().item() + data.test_mask.sum().item() + data.val_mask.sum().item() - data.y.sum().item())

    # create the splits in case of drug target genes, let's remove the positive disease genes from the potential negative genes
    y_train, train_mask, y_test, test_mask, y_val, val_mask = train_test_val_split(node_gene_list=ppi_genes,
                                                                                potential_host_factors_from_publications=potential_drug_targets,
                                                                                strong_host_factors=drug_target_genes,
                                                                                host_factors_shared_from_stukalov_and_others="",
                                                                                validation_set_genes_path="",
                                                                                are_stukalov_shared_genes_used=False,
                                                                                use_all_potential_negatives=True,
                                                                                use_subset_of_negatives=True,
                                                                                fraction_of_negatives_to_select=0.5,
                                                                                is_virus_specific_model=False,
                                                                                is_pretraining_split=True)
    
    data.train_mask_drug_target = torch.tensor(train_mask).bool()
    data.test_mask_drug_target = torch.tensor(test_mask).bool()
    data.val_mask_drug_target = torch.tensor(val_mask).bool()
    data.y_drug_target = torch.tensor(y_val.transpose()[0] + y_test.transpose()[0] + y_train.transpose()[0]).long()
    print("Number of positive labels in drug targets regime: ", data.y_drug_target.sum().item())
    print("Number of negative labels in drug targets regime: ", data.train_mask_drug_target.sum().item() + data.test_mask_drug_target.sum().item() + data.val_mask_drug_target.sum().item() - data.y_drug_target.sum().item())
    
    return data
