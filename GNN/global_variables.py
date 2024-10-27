import yaml
import pandas as pd
import numpy as np
import pickle
from distutils.util import strtobool

#global_variables.storage_folder + 'inference_score_dict_v2.pickle'
import os
class GlobalVariables:
    def __init__(self):
        self.load_config()

    def copy(self):
        # Create a new instance of GlobalVariables
        new_instance = GlobalVariables()
        # Copy all attributes from the current instance to the new instance
        new_instance.__dict__.update(self.__dict__)
        return new_instance
    
    def load_pretraining_genes_from_viral_diseases(self):
        with open(self.config["storage_folder"] + self.config["inference_score_dict_file_name"], "rb") as handle:
            inference_score_dict = pickle.load(handle)
        return inference_score_dict
    
    def load_pretraining_genes_from_all_diseases(self):
        with open(self.config["storage_folder"] + self.config["inference_score_dict_full_file_name"], "rb") as handle:
            inference_score_dict = pickle.load(handle)
        return inference_score_dict
    
    def load_mpxv_data(self):
        # read xlsx file
        df_transcriptome = pd.read_excel(self.config["storage_folder"]  + self.config["mpxv_transcriptomics_file"], sheet_name='A')
        df_proteome = pd.read_excel(self.config["storage_folder"]  + self.config["mpxv_proteomics_file"], sheet_name='A')
        df_host_factors_1 = pd.read_excel(self.config["storage_folder"]  + self.config["mpxv_host_factors_file"], sheet_name='Sheet1', header=5)
        df_host_factors_2 = pd.read_excel(self.config["storage_folder"]  + self.config["mpxv_host_factors_file"], sheet_name='Sheet2', header=4)
        df_host_factors_3 = pd.read_excel(self.config["storage_folder"]  + self.config["mpxv_host_factors_file"], sheet_name='Sheet3', header=5)

        gene_names_1 = df_host_factors_1['Gene symbol'].unique()
        gene_names_2 = df_host_factors_2['GENE_SYMBOL'].unique()
        gene_names_3_1D_1A = df_host_factors_3.groupby('Source of active siRNAs (D, Dharmacon; A, Ambion)')['Gene'].apply(lambda x: x.unique()).loc['1D + 1A']
        gene_names_3_1D_2A = df_host_factors_3.groupby('Source of active siRNAs (D, Dharmacon; A, Ambion)')['Gene'].apply(lambda x: x.unique()).loc['1D + 2A']
        gene_names_3_1D_3A = df_host_factors_3.groupby('Source of active siRNAs (D, Dharmacon; A, Ambion)')['Gene'].apply(lambda x: x.unique()).loc['1D + 3A']
        gene_names_A_Only = df_host_factors_3.groupby('Source of active siRNAs (D, Dharmacon; A, Ambion)')['Gene'].apply(lambda x: x.unique()).loc['A Only']
        host_factors = np.unique(np.concatenate((gene_names_1, gene_names_2, gene_names_3_1D_2A, gene_names_3_1D_3A)))
        weak_host_factors = np.unique(np.concatenate((gene_names_3_1D_1A, gene_names_A_Only)))
        for gene in weak_host_factors:
            if gene in host_factors:
                host_factors = np.delete(host_factors, np.where(host_factors == gene))

        host_factors = np.array([gene.lower() for gene in host_factors])
        weak_host_factors = np.array([gene.lower() for gene in weak_host_factors])
        self.host_factors_mpxv = host_factors
        self.potential_positive_genes_mpxv = np.unique(np.concatenate((host_factors, weak_host_factors)))

        trancriptome_genes = df_transcriptome['gene_name'].dropna().astype(str).unique()
        proteome_genes = df_proteome['gene_names'].dropna().astype(str).unique()
        # duplicate rows in proteome data when the gene names field is composed by multiple genes separated by ; 
        print(f"Accounting for muliple genes corresponding to the same mass spectrometry data.")
        df_proteome_isoforms_filtered = []
        proteome_genes_isoforms_corrected = []
        for gene in proteome_genes:
            row = df_proteome[df_proteome['gene_names'] == gene]
            if ";" in gene:
                #select the ; separated genes
                gene_list = gene.split(";")
                for gene in gene_list:
                    proteome_genes_isoforms_corrected.append(gene)
                    row_to_append = row.copy()
                    row_to_append['gene_names'] = gene
                    df_proteome_isoforms_filtered.append(row_to_append)
            else:
                proteome_genes_isoforms_corrected.append(gene)
                df_proteome_isoforms_filtered.append(row)

        df_proteome_isoforms_filtered = pd.concat(df_proteome_isoforms_filtered, ignore_index=True)
        proteome_genes_isoforms_corrected = np.unique(proteome_genes_isoforms_corrected)

        common_genes = np.unique(np.concatenate((trancriptome_genes, proteome_genes_isoforms_corrected)))
        self.common_genes_mpxv = np.array([gene.lower() for gene in common_genes])
        self.df_transcriptome_mpxv = df_transcriptome
        self.df_proteome_isoforms_filtered_mpxv = df_proteome_isoforms_filtered

    def load_config(self):
        with open(os.getcwd() + "/GNN/config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
        
        self.home_folder = os.getcwd() +"/"
        self.storage_folder = self.config["storage_folder"]
        self.is_dataset_emogi_compatible = self.config["is_dataset_emogi_compatible"]
        self.dataset_type = self.config["dataset_type"]
        self.model_to_use = self.config["model_to_use"]
        self.is_positional_encoding = self.config["is_positional_encoding"]
        self.is_positional_encoding_and_omics = self.config["is_positional_encoding_and_omics"]
        self.positional_encoding_file = self.config["positional_encoding_file"]
        self.gene_names_file = self.config["gene_names_file"]

        host_factors = pd.ExcelFile(os.getcwd() + "/data_preprocessing_pipeline/artivir_data/" + self.config["host_factors_from_publications"])
        host_factors_dfs = {sheet_name: host_factors.parse(sheet_name) for sheet_name in host_factors.sheet_names}
        self.host_factors_from_publications = host_factors_dfs["host_factors"]["Gene name"].str.lower().unique()
        
        self.is_ppr_from_target_to_source = self.config["is_ppr_from_target_to_source"]
        self.is_simple_debug = self.config["is_simple_debug"]
        self.is_pretraining = self.config["is_pretraining"]
        self.is_cross_validation = self.config["is_cross_validation"]
        self.random_seed = self.config["random_seed"]
        self.is_pretraining_on_all_diseases = self.config["is_pretraining_on_all_diseases"]
        self.is_hpo = self.config["is_hpo"]
        self.is_inference = self.config["is_inference"]
        self.is_mlp_used = self.config["is_mlp_used"]
        self.is_pe_used = self.config["is_pe_used"]
        self.is_prot_emb_used = self.config["is_prot_emb_used"]
        self.is_gosemsim_used = self.config["is_gosemsim_used"]
        self.is_omics_used = self.config["is_omics_used"]   

        # training configuration
        training_config = self.config["training_configuration"]
        self.decay = training_config["decay"]
        self.drop_out_rate = training_config["drop_out_rate"]
        self.drop_out_rate_edges = training_config["drop_out_rate_edges"]
        self.hidden_channels_number = training_config["hidden_channels_number"]
        self.hidden_channels_dimension = training_config["hidden_channels_dimension"]
        self.interaction_score_threshold = training_config["interaction_score_threshold"]
        self.latent_space_dimension = training_config["latent_space_dimension"]
        self.jk_mode = training_config["jk_mode"]
        self.learning_rate = training_config["learning_rate"]
        self.n_epochs = training_config["n_epochs"]
        self.negative_weight = training_config["negative_weight"]
        self.positive_weight = training_config["positive_weight"]
        self.is_apply_heat_diffusion = training_config["is_apply_heat_diffusion"]
        self.are_missing_features_imputed = training_config["are_missing_features_imputed"]
        self.n_fine_tuning_steps = training_config["n_fine_tuning_steps"]
        self.is_tune_ppi = training_config["is_tune_ppi"]
        self.n_components = training_config["n_components"]
        self.fine_tuning_virus = training_config["fine_tuning_virus"]

        # validation genes
        self.validated_antiviral_dt = np.array(['gnrhr', 'oprd1', 'chrm3', 'hrh4', 'htr3b', 'itgav', 'kcnh2',
                                                'thra', 'oprk1', 'ntrk2', 'slc6a1', 'kit', 'htr6', 'scnn1g',
                                                'chrm1', 'scn11a', 'htr1e', 'f2', 'htr3c', 'htr1f', 'kcnh7',
                                                'esr2', 'gsto2', 'scnn1b', 'scnn1a', 'htr4', 'grin3a', 'nr1i3',
                                                'f2', 'gsto1', 'abl1', 'scnn1d', 'shbg', 'htr3d', 'chrm5', 'kcnq3',
                                                'noxo1', 'gstp1', 'pomc', 'ppara', 'gabrr2', 'oprm1', 'dhfr',
                                                'chrm2', 'htr3e', 'chrm4', 'scn5a', 'gabrr1', 'mttp', 'ntrk1',
                                                'itgb3', 'kcnh6', 'gabrr3'])

    def add_field(self, field_name, value):
        setattr(self, field_name, value)

    def update_field(self, field_name, value):
        if hasattr(self, field_name):
            setattr(self, field_name, value)
        else:
            raise AttributeError(f"{field_name} does not exist.")

# Create an instance of the class
global_vars = GlobalVariables()
# insert an arg parser to dinamically change the global variables when needed
from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--is_pretraining", type=lambda x: bool(strtobool(x)), default=global_vars.is_pretraining)
    parser.add_argument("--is_cross_validation", type=lambda x: bool(strtobool(x)), default=global_vars.is_cross_validation)
    parser.add_argument("--is_hpo", type=lambda x: bool(strtobool(x)), default=global_vars.is_hpo)
    parser.add_argument("--is_inference", type=lambda x: bool(strtobool(x)), default=global_vars.is_inference)
    parser.add_argument("--is_pretraining_on_all_diseases", type=lambda x: bool(strtobool(x)), default=global_vars.is_pretraining_on_all_diseases)
    parser.add_argument("--is_simple_debug", type=lambda x: bool(strtobool(x)), default=global_vars.is_simple_debug)
    parser.add_argument("--is_apply_heat_diffusion", type=lambda x: bool(strtobool(x)), default=global_vars.is_apply_heat_diffusion)
    parser.add_argument("--are_missing_features_imputed", type=lambda x: bool(strtobool(x)), default=global_vars.are_missing_features_imputed)
    parser.add_argument("--is_tune_ppi", type=lambda x: bool(strtobool(x)), default=global_vars.is_tune_ppi)
    parser.add_argument("--fine_tuning_virus", type=str, default=global_vars.fine_tuning_virus)
    parser.add_argument("--random_seed", type=int, default=global_vars.random_seed)
    parser.add_argument("--is_mlp_used", type=lambda x: bool(strtobool(x)), default=global_vars.is_mlp_used)
    parser.add_argument("--model_to_use", type=str, default=global_vars.model_to_use)
    parser.add_argument("--is_pe_used", type=lambda x: bool(strtobool(x)), default=global_vars.is_pe_used)
    parser.add_argument("--is_prot_emb_used", type=lambda x: bool(strtobool(x)), default=global_vars.is_prot_emb_used)
    parser.add_argument("--is_gosemsim_used", type=lambda x: bool(strtobool(x)), default=global_vars.is_gosemsim_used)
    parser.add_argument("--is_omics_used", type=lambda x: bool(strtobool(x)), default=global_vars.is_omics_used)


    return parser.parse_args()

def update_global_vars(args):
    global_vars.update_field("is_pretraining", args.is_pretraining)
    global_vars.update_field("is_cross_validation", args.is_cross_validation)
    global_vars.update_field("is_hpo", args.is_hpo)
    global_vars.update_field("is_inference", args.is_inference)
    global_vars.update_field("is_pretraining_on_all_diseases", args.is_pretraining_on_all_diseases)
    global_vars.update_field("is_simple_debug", args.is_simple_debug)
    global_vars.update_field("is_apply_heat_diffusion", args.is_apply_heat_diffusion)
    global_vars.update_field("are_missing_features_imputed", args.are_missing_features_imputed)
    global_vars.update_field("is_tune_ppi", args.is_tune_ppi)
    global_vars.update_field("fine_tuning_virus", args.fine_tuning_virus)
    global_vars.update_field("random_seed", args.random_seed)
    global_vars.update_field("is_mlp_used", args.is_mlp_used)
    global_vars.update_field("model_to_use", args.model_to_use)
    global_vars.update_field("is_pe_used", args.is_pe_used)
    global_vars.update_field("is_prot_emb_used", args.is_prot_emb_used)
    global_vars.update_field("is_gosemsim_used", args.is_gosemsim_used)
    global_vars.update_field("is_omics_used", args.is_omics_used)

args = parse_args()
update_global_vars(args)

