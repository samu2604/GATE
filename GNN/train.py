import torch
import numpy as np
import sys, os
from global_variables import global_vars as global_variables
sys.path.append(os.path.abspath(global_variables.home_folder + 'GhostFreePro/data_preprocessing_pipeline'))
sys.path.append(os.path.abspath(global_variables.home_folder + 'GhostFreePro/GNN'))
from torch_geometric.data.data import Data 
from utils import *
from sklearn.metrics import auc, precision_recall_curve, roc_curve, matthews_corrcoef
import models  
from data_preprocessing_pipeline.feature_preselection_transcriptome_proteome import run_artivir_data_preparation, run_pretraining_data_preparation
import torch.nn.functional as F
import random
import optuna
import datetime
import torch.utils.tensorboard as tb
from GNN.dropout import dropout_edge
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer
import tqdm 

data_file_path = global_variables.home_folder + "GhostFreePro/data_preprocessing_pipeline/artivir_data/"

def import_artivir_dataset(data_path, omic_specific_type : str = None, is_randomized_features : bool = False):
    data = load_hdf_data(data_path)
    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    
    # make features trivial for testing
    #features, _ = create_trivial_features(features, y_train, train_mask, y_test, test_mask, y_val, val_mask, "")
    
    if is_randomized_features:
        print("Randomize input features")
        features = features[np.random.permutation(features.shape[0]),:]
        features = features[:, np.random.permutation(features.shape[1])]
    
    pytorch_data = Data()
    pytorch_data.edge_index = torch.Tensor(np.where(network != 0)).long()
    pytorch_data.edge_weight = torch.Tensor(network[np.where(network != 0)])
    pytorch_data.num_node_features = 0
    if "transcriptome" in omic_specific_type:    
        pytorch_data.num_node_features += torch.tensor(np.shape(features[:,0:3])[1]).long()
    if "proteome" in omic_specific_type:
        pytorch_data.num_node_features += torch.tensor(np.shape(features[:,3:6])[1]).long()
    if "effectome" in omic_specific_type:
        pytorch_data.num_node_features += torch.tensor(np.shape(features[:,6:30])[1]).long()
    if "interactome" in omic_specific_type:
        pytorch_data.num_node_features += torch.tensor(np.shape(features[:,30:54])[1]).long()

    print("Use omic type " + omic_specific_type)
    pytorch_data.x = torch.tensor(features).float()
    
    print(f"Number of node features {pytorch_data.num_node_features}")
    pytorch_data.y = torch.tensor(y_val.transpose()[0] + y_test.transpose()[0] + y_train.transpose()[0]).long()
    pytorch_data.train_mask = torch.tensor(train_mask).bool()
    pytorch_data.test_mask = torch.tensor(test_mask).bool()
    pytorch_data.val_mask = torch.tensor(val_mask).bool()
    
    return pytorch_data


def train(model, optimizer, data, loss_func, drop_out_rate, drop_out_rate_edges, masks_list):
    model.train()
    optimizer.zero_grad() 

    concat_features_matrix = torch.cat((data.x, data.x_ft), dim = 1)
    concat_features_matrix.requires_grad = True
    out, out_dt = model(concat_features_matrix, data.edge_index, data.edge_weight, drop_out_rate, drop_out_rate_edges)
    
    if model.is_gnn_explainer:  
        model.eval()

        model_explainer = models.ModelExplainer(model, data, drop_out_rate, drop_out_rate_edges, model.device, is_drug_targets_output = False, fine_tuning_virus = global_variables.fine_tuning_virus)
        model_explainer_dt = models.ModelExplainer(model, data, drop_out_rate, drop_out_rate_edges, model.device, is_drug_targets_output = True, fine_tuning_virus = global_variables.fine_tuning_virus)
        model_explainer.training = model.training
        number_of_explaining_epochs = 100
        print(f"Number of explaining epochs {number_of_explaining_epochs}")
        explainer = Explainer(
            model=model_explainer, 
            algorithm=GNNExplainer(epochs=number_of_explaining_epochs), #GraphMaskExplainer(num_layers=2, epochs=10), 
            explanation_type='phenomenon',
            node_mask_type="attributes",
            edge_mask_type="object",
                model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )
        explainer_dt = Explainer(
            model=model_explainer_dt, 
            algorithm=GNNExplainer(epochs=number_of_explaining_epochs), #GraphMaskExplainer(num_layers=2, epochs=10), 
            explanation_type='phenomenon',
            node_mask_type="attributes",
            edge_mask_type="object",
                model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='probs',
            ),
        )

        with torch.no_grad():
            k = 100
            print(f"Run explanations for top {k} nodes")
            pos_indices = torch.where(data.y == 1)[0]
            pos_indices_dt = torch.where(data.y_drug_target == 1)[0]
            print(f"Number of positive indices for dg and dt are {data.y.sum()} and {data.y_drug_target.sum()}")
            probs = model_explainer(concat_features_matrix, data.edge_index, data.edge_weight)
            probs_dt = model_explainer_dt(concat_features_matrix, data.edge_index, data.edge_weight)
            top_k_nodes = pos_indices[torch.argsort(probs[pos_indices], descending=True)[:k]]
            top_k_nodes_dt = pos_indices_dt[torch.argsort(probs_dt[pos_indices_dt], descending=True)[:k]]
            subgraph_masks = []
            feature_mask = None
            
        for index, index_dt in zip(top_k_nodes, top_k_nodes_dt):  
            print(f"Explaining disease and drug target nodes {index} and {index_dt} with GNNExplainer and positive probs {probs[index]} and {probs_dt[index_dt]}")              
            explanation = explainer(concat_features_matrix, data.edge_index, target=data.y, edge_weight=data.edge_weight, index=index)
            explanation_dt = explainer_dt(concat_features_matrix, data.edge_index, target=data.y_drug_target, edge_weight=data.edge_weight, index=index_dt)
            subgraph_masks.append(explanation.edge_mask)
            if feature_mask is None:
                feature_mask = explanation.node_mask
            else:
                feature_mask += explanation.node_mask  
            subgraph_masks.append(explanation_dt.edge_mask)
            feature_mask += explanation_dt.node_mask

        feature_mask = (feature_mask.max(dim=0)[0]/feature_mask.max(dim=0)[0].max())
        omic_features_size = model.omics_features_matrix.size(1)
        pe_features_size = model.input_pe_dimension_size
        go_feature_size = model.input_gosemsim_dimension_size
        prot_embeddings_size = model.input_prot_emb_matrix.size(1)
        # print the mean and max of the feature mask for pe go and prot embeddings separately
        pe_mean_importance = feature_mask[:pe_features_size].mean()
        pe_max_importance = feature_mask[:pe_features_size].max()
        go_mean_importance = feature_mask[pe_features_size:pe_features_size + go_feature_size].mean()
        go_max_importance = feature_mask[pe_features_size:pe_features_size + go_feature_size].max()
        prot_emb_mean_importance = feature_mask[pe_features_size + go_feature_size:pe_features_size + go_feature_size + prot_embeddings_size].mean()
        prot_emb_max_importance = feature_mask[pe_features_size + go_feature_size:pe_features_size + go_feature_size + prot_embeddings_size].max()
        print(f"Mean and max importance for pe {pe_mean_importance} and {pe_max_importance}")
        print(f"Mean and max importance for go {go_mean_importance} and {go_max_importance}")
        print(f"Mean and max importance for prot emb {prot_emb_mean_importance} and {prot_emb_max_importance}")
        means = np.array([pe_mean_importance.item(), go_mean_importance.item(), prot_emb_mean_importance.item()])
        features_importance_threshold = np.mean(means) - np.std(means)
        features_importance_threshold = features_importance_threshold.item() if features_importance_threshold.item() > 0 else 0
        print(f"Feature importance threshold {features_importance_threshold}")
        print(f"fraction of pe features with importance above threshold {torch.where(feature_mask[:pe_features_size] > features_importance_threshold)[0].size(0)/pe_features_size}")
        print(f"fraction of go features with importance above threshold {torch.where(feature_mask[pe_features_size:pe_features_size + go_feature_size] > features_importance_threshold)[0].size(0)/go_feature_size}")
        print(f"fraction of prot emb features with importance above threshold {torch.where(feature_mask[pe_features_size + go_feature_size:pe_features_size + go_feature_size + prot_embeddings_size] > features_importance_threshold)[0].size(0)/prot_embeddings_size}")
        # avoid selecting omic features that are still not used
        unimportant_features_indices = torch.where(feature_mask[:-omic_features_size] < features_importance_threshold)[0]
        print(f"Number of unimportant features {len(unimportant_features_indices)}")
        print("set the unimportant features to 0 in concat_features_matrix and reset model.feature_matrix for pe, gosemsim and prot embeddings")
        # Temporarily disable gradient tracking to perform the in-place operation, afterwards gradient-tracking is automatically re-enabled
        if len(unimportant_features_indices) > 0:
            with torch.no_grad():
                concat_features_matrix[:, unimportant_features_indices] = 0
            model.reset_features_matrix_values(concat_features_matrix)

        max_edge_mask = torch.stack(subgraph_masks).max(dim=0)[0]
        # Normalise weights
        max_edge_maks_min = max_edge_mask[max_edge_mask > 0].min()
        # create tensor with values max_edge_maks_min if max_edge_mask is > 0 and 0 otherwise
        max_edge_maks_min_tensor = torch.where(max_edge_mask > 0, max_edge_maks_min, 0)
        max_edge_mask = (max_edge_mask - max_edge_maks_min_tensor) / (
            max_edge_mask.max() - max_edge_maks_min_tensor
        )
        new_edge_weight = max_edge_mask
        # Update edge weights and remove edges with weight below 0.9 that are not selected as important by the explainer
        def find_threshold_for_top_k(edge_weights, k=250000):
            print(f"threshold for top k {k}")
            print(f"Max, min, mean and std of edge weights {edge_weights.max()}, {edge_weights.min()}, {edge_weights.mean()}, {edge_weights.std()}")
            print(f"Number of edges > 0.0 {torch.where(torch.tensor(edge_weights) > 0.0)[0].size(0)}")
            # Sort the edge weights in descending order
            sorted_weights = np.sort(edge_weights)[::-1]
            # Find the threshold corresponding to the k-th largest edge
            threshold = sorted_weights[k - 1]
            return threshold
        print(f"find threshold for edge weights")
        threshold_input_ppi = find_threshold_for_top_k(data.edge_weight.cpu().numpy(), k=300000)
        print(f"find threshold for explained edges")
        threshold_explained_edges = find_threshold_for_top_k(new_edge_weight.cpu().numpy(), k=200000)
        print(f"Thresholds for input ppi and explained edges {threshold_input_ppi} and {threshold_explained_edges}")

        edges_to_keep_mask = ((data.edge_weight > threshold_input_ppi) | (new_edge_weight > threshold_explained_edges))
        # keep the maximum weight between the two edge weights at the same edge index
        data.edge_weight = torch.max(data.edge_weight[edges_to_keep_mask], new_edge_weight[edges_to_keep_mask])
        data.edge_index = data.edge_index[:, edges_to_keep_mask]
        data.num_nodes = len(torch.unique(data.edge_index))

        print(f"Number of edges after GNNExplainer {data.edge_index.size()[1]}")
        # save the new data obj containing edge weights and edge index to file for further analysis with a unique name timestamp
        torch.save(data, global_variables.storage_folder + "data_explained_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".pt")
        model.is_gnn_explainer = False

        # # start tuning the message weights for the remaining connections in the graph
        # model.init_message_weights(data)
        # # update the optimizer to include the message weights
        # optimizer = torch.optim.Adam(model.parameters(), 
        #                     lr=optimizer.param_groups[0]['lr'],
        #                     weight_decay=optimizer.param_groups[0]['weight_decay'])
        # model.is_message_weights_used = True
        model.train()
                

    # Only use nodes with labels available for loss calculation --> mask
    # select a different set of negatives at each epoch
    if model.is_fine_tuning:
        train_mask_dg = return_new_mask(data.train_mask_ft.cpu().clone().numpy(), data, 3, is_drug_target = False, is_fine_tuning = True)
    else:
        train_mask_dg = return_new_mask(data.train_mask.cpu().clone().numpy(), data, 1, is_drug_target = False) 

    test_mask_dg = return_new_mask(data.test_mask_ft.cpu().clone().numpy(), data, 3, is_drug_target = False, is_fine_tuning = True)

    train_mask_dt = return_new_mask(data.train_mask_drug_target.cpu().clone().numpy(), data, 1, is_drug_target = True)
    test_mask_dt = return_new_mask(data.test_mask_drug_target.cpu().clone().numpy(), data, 1, is_drug_target = True)
    loss = loss_func(out, out_dt, data, train_mask_dg, train_mask_dt)
    criterion_test_pt = torch.nn.CrossEntropyLoss(weight=torch.tensor([global_variables.negative_weight, global_variables.positive_weight], device=data.x.device))
    test_loss = loss_func(out, out_dt, data, test_mask_dg, test_mask_dt, dg_targets=data.y_ft, criterion_dg=criterion_test_pt) if not model.is_fine_tuning else loss_func(out, out_dt, data, test_mask_dg, test_mask_dt)
    masks = Masks(train_mask=train_mask_dg, test_mask=test_mask_dg,train_mask_dt=train_mask_dt, test_mask_dt=test_mask_dt)
    masks_list.append(masks) 

    #TODO: is it necessary to average the loss over the number of masks for the fine-tuning?
    if len(masks_list) >= 3 and model.is_fine_tuning:
        for mask in masks_list[-3:-1]:
            loss += loss_func(out, out_dt, data, mask.train_mask, mask.train_mask_dt)
            test_loss += loss_func(out, out_dt, data, mask.test_mask, mask.test_mask_dt)
        loss = loss/3
        test_loss = test_loss/3
    
    #previous_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
    loss.backward() 
    optimizer.step() 
    #check_updated_weights(model, previous_params)
    
    return loss, test_loss



def compute_metrics(set_mask, pred, labels):
    prediction_probs = pred
    metrics = {}
    
    precision_values, recall_values, aupr_thresholds = precision_recall_curve(y_true=labels[set_mask].detach().cpu().numpy(), \
                                                                                     probas_pred=prediction_probs[set_mask].detach().cpu().numpy())
    fpr, tpr, auroc_thresholds = roc_curve(y_true=labels[set_mask].detach().cpu().numpy(), \
                                                                                        y_score=prediction_probs[set_mask].detach().cpu().numpy())
    metrics["aupr"] = auc(recall_values, precision_values)
    metrics["auroc"] = auc(fpr, tpr)  

    # Find the threshold that maximizes the harmonic mean of precision and recall
    sum_precision_recall = precision_values + recall_values
    # set f1 score to 0 where sum_precision_recall is 0 to avoid division by zero
    sum_precision_recall[sum_precision_recall == 0] = 1
    f1_scores = 2 * (precision_values * recall_values) / sum_precision_recall
    best_threshold_index = f1_scores.argmax()
    best_threshold = aupr_thresholds[best_threshold_index]
    
    pred = torch.where(pred > best_threshold, 1, 0)
    metrics["accuracy"] = accuracy(pred[set_mask], labels[set_mask]) if labels[set_mask].sum() > 0 else 0
    metrics["true_positives"] = true_positive(pred = pred[set_mask], target = labels[set_mask], positive_class = 1)
    metrics["true_negatives"] = true_negative(pred = pred[set_mask], target = labels[set_mask], negative_class = 0)
    metrics["false_positives"] = false_positive(pred = pred[set_mask], target = labels[set_mask], positive_class = 1)
    metrics["false_negatives"] = false_negative(pred = pred[set_mask], target = labels[set_mask], negative_class = 0)
    metrics["precision_val"] = precision(pred = pred[set_mask], target = labels[set_mask], positive_class = 1)
    metrics["recall_val"] = recall(pred = pred[set_mask], target = labels[set_mask])
    metrics["f1_score_val"] = f1_score(pred = pred[set_mask], target = labels[set_mask]) 
    metrics["best_classification_threshold"] = best_threshold
    metrics["MCC"] = matthews_corrcoef(y_pred = pred[set_mask].detach().cpu().numpy(), y_true = labels[set_mask].detach().cpu().numpy())
    
    return metrics

def binary_cross_entropy_with_logits_wrapper(pos_weight):
    def binary_cross_entropy_with_logits_wrapper(logits, target):
        return torch.nn.functional.binary_cross_entropy_with_logits(input = logits, target = target.float(), pos_weight = pos_weight)
    return binary_cross_entropy_with_logits_wrapper


def test(model, data, masks):
    model.eval()
    
    concat_features_matrix = torch.cat((data.x, data.x_ft), dim = 1)
    out_dg, out_dt = model(concat_features_matrix, data.edge_index, data.edge_weight, 0.0, 0.0) 

    if isinstance(model, models.MPNN):
        pred_dg = F.softmax(out_dg, dim=1)[:,1]  
        pred_dt = F.softmax(out_dt, dim=1)[:,1] 
        train_mask = masks.train_mask
        train_mask_dt = masks.train_mask_dt 
        test_mask = masks.test_mask
        test_mask_dt = masks.test_mask_dt 
        labels = data.clone().cpu().y if not model.is_fine_tuning else data.clone().cpu().y_ft
        train_set_metrics = compute_metrics(train_mask, pred_dg.clone().cpu(), labels)
        if not model.is_fine_tuning: # if pre-training, use the fine-tuning labels for the test metrics of disease genes head
            labels = data.clone().cpu().y_ft
        test_set_metrics = compute_metrics(test_mask, pred_dg.clone().cpu(), labels)
        train_set_metrics_dt = compute_metrics(train_mask_dt, pred_dt.clone().cpu(), data.clone().cpu().y_drug_target)
        test_set_metrics_dt = compute_metrics(test_mask_dt, pred_dt.clone().cpu(), data.clone().cpu().y_drug_target)
        #   if model.is_fine_tuning:
        #       print(f"len global_variables.validated_antiviral_dt {len(global_variables.validated_antiviral_dt)}")
        #       global_variables.validated_antiviral_dt, data = check_if_drug_targets_are_classified_as_disease_genes(data, pred_dg.clone().cpu(), global_variables.validated_antiviral_dt, counter_max=5, is_used_fixed_list=False)
        #       print(f"len global_variables.validated_antiviral_dt {len(global_variables.validated_antiviral_dt)}")

        return train_set_metrics, test_set_metrics, train_set_metrics_dt, test_set_metrics_dt, pred_dg, pred_dt

def positive_negatives_splitter(mask, y_drug_target_mask, y_disease_genes_mask, is_drug_target):
    if is_drug_target:
        negatives_mask = mask^(y_drug_target_mask)& mask
    else:
        negatives_mask = mask^(y_disease_genes_mask)& mask
    positives_mask = negatives_mask^mask
    return negatives_mask, positives_mask

def return_new_mask(mask, data, imbalance_factor: int, is_drug_target: bool = False, is_fine_tuning: bool = False):
    y_drug_target_mask = data.y_drug_target.cpu().numpy().astype(bool)
    y_disease_genes_mask = data.y.cpu().numpy().astype(bool) if not is_fine_tuning else data.y_ft.cpu().numpy().astype(bool)
    negatives_mask, positives_mask = positive_negatives_splitter(mask, y_drug_target_mask, y_disease_genes_mask, is_drug_target)
    number_of_positives = np.sum(positives_mask)
    negative_indices = np.where(negatives_mask == True)[0]
    positive_indices = np.where(positives_mask == True)[0]
    if number_of_positives < int(len(negative_indices)/imbalance_factor):
        negative_indices_selection = random.sample(sorted(negative_indices), 
                                               number_of_positives*imbalance_factor)
        positive_indices_selection = positive_indices
    else:
        positive_indices_selection = random.sample(sorted(positive_indices), int(len(negative_indices)/imbalance_factor))
        positive_mask = np.zeros(len(positives_mask)).astype(bool)
        positive_mask[positive_indices_selection] = True
        negative_indices_selection = negative_indices

    new_mask = np.zeros(len(negatives_mask)).astype(bool)
    new_mask[negative_indices_selection] = True
    new_mask[positive_indices_selection] = True 
    return new_mask

def impute_zero_values_with_column_mean(data):
    """
    Impute all the 0.0 values in data.x with the average of the corresponding column,
    excluding zeros in the calculation of the mean.
    
    Args:
        data: A Data object containing the 'x' attribute which is a PyTorch tensor.
    """
    print("Impute zero values with column mean")
    # For each column, calculate the mean of non-zero elements and replace 0s
    for col in range(data.x.shape[1]):
        column_values = data.x[:, col]
        
        # Get non-zero values from the column
        non_zero_values = column_values[column_values != 0.0]
        
        # If there are non-zero values, calculate the mean
        if len(non_zero_values) > 0:
            column_mean = torch.mean(non_zero_values)
            
            # Replace 0s in the original column with the calculated mean
            data.x[:, col] = torch.where(column_values == 0.0, column_mean, column_values)
    

def initialiaze_model(data : Data, is_latent_space_summed: bool, hidden_channels: list, apply_heat_diffusion: bool = False, jk_mode: str = "cat"):
    # inpute zero values with column mean
    #if global_variables.are_missing_features_imputed:
    #    impute_zero_values_with_column_mean(data)
    if apply_heat_diffusion == True and global_variables.is_pretraining:     
        print("Applying heat diffusion to omics features")
        data.x_ft = models.heat_diffusion(edge_index=data.edge_index, edge_weight=data.edge_weight, features_matrix=data.x_ft, num_nodes=data.x_ft.size()[0])
  
    model_to_use = models.MPNN
    model = model_to_use(data, hidden_channels, jk_mode = jk_mode, input_pe_dimension_size = global_variables.n_components, input_gosemsim_dimension_size = global_variables.n_components, fine_tuning_virus = global_variables.fine_tuning_virus)
        
    return model

def prepare_data(artivir_data_dir, are_missing_features_imputed, interaction_score_threshold, binary_compound_score_values):
    is_used_abs_value_for_up_down_regulated = False
    input_graph_type = "STRING"
    
    strong_host_factors = artivir_data_dir + "strong_functionally_validated_host_factors.npy"
    host_factors_shared_from_stukalov_and_others = artivir_data_dir + "host_factors_stukalov_plus_two_more_studies.txt"
    is_trivial_features = False
    are_stukalov_shared_genes_used = False
    hdf5_file_name = global_variables.dataset_type
    validation_set_genes_path = global_variables.storage_folder + "anti_viral_drug_targets.txt"#artivir_data_dir + "validation_gene_list.txt"
    disease_genes_related_to_viral_infections = global_variables.storage_folder + "positive_selected_genes.txt"

    return run_artivir_data_preparation(artivir_data_dir, strong_host_factors, 
                                   global_variables.dataset_type, host_factors_shared_from_stukalov_and_others,
                                   is_trivial_features, hdf5_file_name, is_used_abs_value_for_up_down_regulated,
                                   are_missing_features_imputed, input_graph_type, are_stukalov_shared_genes_used,
                                   is_append_zero_imputed_features = True,
                                   interaction_score_threshold = interaction_score_threshold,
                                   validation_set_genes_path = validation_set_genes_path,
                                   binary_compound_score_values = binary_compound_score_values,
                                   disease_genes_related_to_viral_infections = disease_genes_related_to_viral_infections)                                                                                               


def prepare_pretraining_data(data_ft):
    storage_folder = global_variables.storage_folder + "input_pretraining/"
    pe_encoding_ranks = np.load(storage_folder + global_variables.positional_encoding_file, allow_pickle=True)
    pe_encodings = run_positional_encoding_dimensionality_reduction(ranks=pe_encoding_ranks, n_components=global_variables.n_components)
    gene_names_ppi_ppr = np.load(storage_folder + global_variables.gene_names_file, allow_pickle=True)

    df_go_sem_sim_ranks = pd.read_parquet(storage_folder + 'similarity_matrix_avg_reduced.parquet')
    go_sem_sim_ranks = df_go_sem_sim_ranks.to_numpy()
    go_sem_sim_matrix_reduced = run_positional_encoding_dimensionality_reduction(ranks=go_sem_sim_ranks, n_components=global_variables.n_components, is_standardized=False)

    genes_gosemsim = np.loadtxt(storage_folder + 'gosemsim_gene_names.txt', dtype=str)
    
    protein_embeddings_file_path =  storage_folder + "ppi_protein_embeddings.pickle"
    data = run_pretraining_data_preparation(pe_encoding_file=pe_encodings,
                                genes_ppi_pe=gene_names_ppi_ppr, go_sem_sim_matrix_reduced=go_sem_sim_matrix_reduced, \
                                genes_gosemsim=genes_gosemsim, data_ft=data_ft, \
                                protein_embeddings_file_path=protein_embeddings_file_path)
    return data

def prepare_positional_encoding_data(data):
    
    device = data.x.device
    ranks = torch.load(global_variables.storage_folder + 'ppr/' + global_variables.positional_encoding_file)
    ranks = run_positional_encoding_dimensionality_reduction(ranks)
    gene_names = np.load(global_variables.storage_folder + 'ppr/' + global_variables.gene_names_file, allow_pickle=True)

    positional_encoding_features = []
    if global_variables.is_positional_encoding_and_omics:
        data.node_names = data.node_names[:,0]
    for gene in data.node_names:
        if gene in gene_names:
            gene_index = np.where(gene_names == gene)[0][0]
            positional_encoding_features.append(ranks[gene_index,:])
        else:
            print(f"Gene {gene} not found in positional encoding")
            positional_encoding_features.append(torch.zeros(ranks.size()[1]))
    positional_encoding_features = torch.stack(positional_encoding_features, dim = 0).to(device)

    if global_variables.is_positional_encoding:
        print("Use positional encoding")
        data_positional_encoding = Data()
        data_positional_encoding = data.clone('train_mask', 'edge_weight', 'x', 'node_names', 'test_mask', 'val_mask', 'feat_names', 'num_node_features', 'edge_index', 'y')
        data_positional_encoding.x = positional_encoding_features
        data_positional_encoding.num_node_features = 0
        data_positional_encoding.num_node_pe_dimensions = data_positional_encoding.x.size()[1]
        data_positional_encoding = data_positional_encoding.to(device)

    elif global_variables.is_positional_encoding_and_omics:
        print("Use positional encoding and omics")
        data_positional_encoding = Data()
        data_positional_encoding = data.clone('train_mask', 'edge_weight', 'node_names', 'test_mask', 'val_mask', 'feat_names', 'num_node_features', 'edge_index', 'y')
        data_positional_encoding.x = torch.cat((data.x, positional_encoding_features), dim = 1)
        data_positional_encoding.num_node_features = data.x.size()[1]
        data_positional_encoding.num_node_pe_dimensions = data_positional_encoding.x.size()[1] - data.x.size()[1]
        data_positional_encoding = data_positional_encoding.to(device)
        
    data = data_positional_encoding.clone()
    del data_positional_encoding  

    return data

def get_trial_parameters(trial):
    
    is_tune_ppi = global_variables.is_tune_ppi
    apply_heat_diffusion = global_variables.is_apply_heat_diffusion
    are_missing_features_imputed = global_variables.are_missing_features_imputed
    if not global_variables.is_inference:
        decay = global_variables.decay[0] if is_tune_ppi else trial.suggest_float('decay', global_variables.decay[0], global_variables.decay[1]) 
        drop_out_rate = global_variables.drop_out_rate[0]  if is_tune_ppi else trial.suggest_float('drop_out_rate', global_variables.drop_out_rate[0], global_variables.drop_out_rate[1])
        drop_out_rate_edges = global_variables.drop_out_rate_edges[0] if is_tune_ppi else trial.suggest_float('drop_out_rate_edges', global_variables.drop_out_rate_edges[0], global_variables.drop_out_rate_edges[1])
        hidden_channels_number = global_variables.hidden_channels_number[0] if is_tune_ppi else trial.suggest_int('hidden_channels_number', global_variables.hidden_channels_number[0], global_variables.hidden_channels_number[1])
        hidden_channels_dimension = global_variables.hidden_channels_dimension[0] if is_tune_ppi else trial.suggest_int('hidden_channels_dimension', global_variables.hidden_channels_dimension[0], global_variables.hidden_channels_dimension[1])
        hidden_channels = [hidden_channels_dimension]*hidden_channels_number
        print(f"hidden_channels {hidden_channels}")
        interaction_score_threshold = global_variables.interaction_score_threshold[0] if is_tune_ppi else trial.suggest_float('interaction_score_threshold', global_variables.interaction_score_threshold[0], global_variables.interaction_score_threshold[1])
        is_latent_space_summed = True if is_tune_ppi else trial.suggest_categorical('is_latent_space_summed', [True])
        jk_mode = global_variables.jk_mode[0] #if is_tune_ppi else trial.suggest_categorical('jk_mode', global_variables.jk_mode)
        learning_rate = global_variables.learning_rate[0] if is_tune_ppi else trial.suggest_float('learning_rate', global_variables.learning_rate[0], global_variables.learning_rate[1]) 
        n_epochs = global_variables.n_epochs[0] if is_tune_ppi else trial.suggest_int('n_epochs', global_variables.n_epochs[0], global_variables.n_epochs[1])
        latent_space_omic_dimension = global_variables.latent_space_dimension[0] if is_tune_ppi else trial.suggest_int('latent_space_omic_dimension', global_variables.latent_space_dimension[0], global_variables.latent_space_dimension[1])
        n_fine_tuning_steps = global_variables.n_fine_tuning_steps[0] if is_tune_ppi else trial.suggest_int('n_fine_tuning_steps', global_variables.n_fine_tuning_steps[0], global_variables.n_fine_tuning_steps[1])
        print(f"n_fine_tuning_steps {n_fine_tuning_steps}")
    else:
        # dict_keys(['decay', 'drop_out_rate', 'drop_out_rate_edges', 'hidden_channels_number', 'hidden_channels_dimension', 
        # 'interaction_score_threshold', 'learning_rate', 'n_epochs', 'latent_space_omic_dimension', 'n_fine_tuning_steps'])
        best_params = global_variables.best_params
        decay = best_params['decay']
        drop_out_rate = best_params['drop_out_rate']
        drop_out_rate_edges = best_params['drop_out_rate_edges']
        hidden_channels_number = best_params['hidden_channels_number']
        hidden_channels_dimension = best_params['hidden_channels_dimension']
        hidden_channels = [hidden_channels_dimension]*hidden_channels_number
        print(f"hidden_channels {hidden_channels}")
        interaction_score_threshold = best_params['interaction_score_threshold']
        is_latent_space_summed = True
        jk_mode = global_variables.jk_mode[0]
        learning_rate = best_params['learning_rate']
        n_epochs = best_params['n_epochs']
        latent_space_omic_dimension = best_params['latent_space_omic_dimension']
        n_fine_tuning_steps = best_params['n_fine_tuning_steps']
        print(f"n_fine_tuning_steps {n_fine_tuning_steps}")
        # log global_variables.random_seed
        print(f"Random seed {global_variables.random_seed}")
        trial.set_user_attr('random_seed', global_variables.random_seed)
    
    #list_length = 12
    #binary_compound_score_values = [1 for i in range(list_length)]
    binary_compound_score_values = [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1] # winning combination
    #binary_compound_score_values = [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1] # tuned for PPI
    #binary_compound_score_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # good
    #binary_compound_score_values = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    #binary_compound_score_values = [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    #binary_compound_score_values = [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    #binary_compound_score_values = [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    #binary_compound_score_values = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1] # good
    #binary_compound_score_values = [1, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 1, 1]
    #binary_compound_score_values = [0.5, 1, 0.5, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1]
    
    #binary_compound_score_values = [trial.suggest_int('binary_compound_score_0', 0, 1), trial.suggest_int('binary_compound_score_1', 0, 1), 0 , trial.suggest_int('binary_compound_score_3', 0, 1), trial.suggest_int('binary_compound_score_4', 0, 1), trial.suggest_int('binary_compound_score_5', 0, 1), \
    #                                trial.suggest_int('binary_compound_score_6', 0, 1), trial.suggest_int('binary_compound_score_7', 0, 1), trial.suggest_int('binary_compound_score_8', 0, 1), trial.suggest_int('binary_compound_score_9', 0, 1), trial.suggest_int('binary_compound_score_10', 0, 1), trial.suggest_int('binary_compound_score_11', 0, 1)]
    print(f"proposed binary_compound_score_values {binary_compound_score_values}")
    #if is_tune_ppi and global_variables.is_hpo:
    #    print("Tune PPI")
    #    binary_compound_score_values = [trial.suggest_int(f'binary_compound_score_{i}', 0, 1) for i in range(list_length)]
    print(f"binary_compound_score_values {binary_compound_score_values}")
    negative_weight = global_variables.negative_weight
    positive_weight = global_variables.positive_weight
    artivir_data_dir = global_variables.home_folder + "data_preprocessing_pipeline/artivir_data/"
    n_folds = 10 if global_variables.is_cross_validation else 1

    return apply_heat_diffusion, are_missing_features_imputed, decay, drop_out_rate, drop_out_rate_edges, \
        hidden_channels, interaction_score_threshold, is_latent_space_summed, jk_mode, learning_rate, n_epochs, \
            negative_weight, positive_weight, artivir_data_dir, n_folds, latent_space_omic_dimension, n_fine_tuning_steps, binary_compound_score_values

def create_cv_data(data, k_folds, cv_run, k_folds_dt, k_folds_ft):
    cv_data = Data()
    cv_data.node_names = data.node_names

    cv_data.train_mask = torch.tensor(k_folds[cv_run][0]).bool()
    cv_data.test_mask = torch.tensor(k_folds[cv_run][1]).bool()

    cv_data.train_mask_drug_target = torch.tensor(k_folds_dt[cv_run][0]).bool()
    cv_data.test_mask_drug_target = torch.tensor(k_folds_dt[cv_run][1]).bool()
    cv_data.y_drug_target = data.y_drug_target.clone()

    cv_data.train_mask_ft = torch.tensor(k_folds_ft[cv_run][0]).bool()
    cv_data.test_mask_ft = torch.tensor(k_folds_ft[cv_run][1]).bool()
    cv_data.x_ft = data.x_ft.clone()
    cv_data.y_ft = data.y_ft.clone()

    cv_data.edge_weight = data.edge_weight.clone()
    cv_data.edge_index = data.edge_index.clone()
    cv_data.x = data.x.clone()
    cv_data.y = data.y.clone()

    return cv_data

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial=None):
    #set random seeds
    set_seed(global_variables.random_seed)

    apply_heat_diffusion, are_missing_features_imputed, decay, drop_out_rate, \
        drop_out_rate_edges, hidden_channels, interaction_score_threshold, \
            is_latent_space_summed, jk_mode, learning_rate, n_epochs, negative_weight, \
                positive_weight, artivir_data_dir, n_folds, latent_space_omic_dimension, n_fine_tuning_steps, binary_compound_score_values = get_trial_parameters(trial) 
    
    assert (n_folds > 1 and global_variables.is_cross_validation == True) or (n_folds == 1 and global_variables.is_cross_validation == False), "Cross validation flag is not consistent with the number of folds"
    
    data_ft = prepare_data(artivir_data_dir, are_missing_features_imputed, interaction_score_threshold, binary_compound_score_values) 
    data = prepare_pretraining_data(data_ft)
    if False:
        data_explained = torch.load(global_variables.storage_folder + "data_explained.pt")
        print(f"Number of edges in explained data: {data_explained.edge_index.size(1)}")
        
        # Set the filtered edge_index from the explained data
        data.edge_index = data_explained.edge_index
        data.edge_weight = torch.ones(data_explained.edge_index.size(1))
        data.num_nodes = len(torch.unique(data_explained.edge_index))

        # If required, apply the edge mask to edge weights or other edge-related tensors
        # For example, if data has edge weights:
        # if hasattr(data, 'edge_weight'):
        #     data.edge_weight = data.edge_weight[edge_mask]
    # rescale the edge weights from 0 to 1
    #data.edge_weight = (data.edge_weight - data.edge_weight.min()) / (data.edge_weight.max() - data.edge_weight.min())


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    if trial is not None:    
        trial.set_user_attr("node_names", data.node_names.tolist())
    
    print(f"Number of node features {data.num_node_features}")
    model = initialiaze_model(data=data, is_latent_space_summed=is_latent_space_summed, hidden_channels=hidden_channels, apply_heat_diffusion = apply_heat_diffusion, jk_mode = jk_mode)
    #add fields to model that are not in the original model but are needed for the explainer run
    model.hidden_channels = hidden_channels
    model.latent_space_omic_dimension = latent_space_omic_dimension

    #assert isinstance(model, models.MPNN) and global_variables.is_pretraining == True or isinstance(model, models.GCN) and global_variables.is_pretraining == False or isinstance(model, models.MLP) and global_variables.is_pretraining == False, 'Wrong is_pretraining flag'
    #DEBUG
    #Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate,
                                weight_decay=decay)
        

    # use a binary cross entropy loss with logits
    imbalance_dataset_weights = torch.tensor([negative_weight, positive_weight], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight = imbalance_dataset_weights) #torch.nn.BCEWithLogitsLoss() 

    model = model.to(device)
    global_variables.validation_performance = ValidationPerformances()
    global_variables.performances = Performances()

    if n_folds >= 2:
        #do CV split
        # implement stratified k-fold for disease related genes (pretraining)
        mask = data.train_mask + data.val_mask 
        k_folds = k_fold(mask.clone().cpu().numpy(), data.y.clone().cpu().numpy(), n_folds)
        k_folds_dt = None
        # implement stratified k-fold for drug targets (pretraining)
        mask_dt = data.train_mask_drug_target + data.val_mask_drug_target 
        k_folds_dt = k_fold(mask_dt.clone().cpu().numpy(), data.y_drug_target.clone().cpu().numpy(), n_folds)
        # implement stratified k-fold for fine tuning 
        mask_ft = data.train_mask_ft + data.val_mask_ft 
        k_folds_ft = k_fold(mask_ft.clone().cpu().numpy(), data.y_ft.clone().cpu().numpy(), n_folds)

        #for name, param in model.named_parameters():
        #    if name not in ["linear_prot_emb_encoder_layer.weight", "linear_prot_emb_encoder_layer.bias", "linear_pe_encoder_layer.weight", "linear_pe_encoder_layer.bias", "linear_gosemsim_encoder_layer.weight", "linear_gosemsim_encoder_layer.bias", "message_weights"]:
        #        print(f"model parameter {name}")
        #        #param.requires_grad = False
        pot_hf_diff = 0
        pot_dt_diff = 0
        initial_antiviral_dt = global_variables.validated_antiviral_dt

        for cv_run in np.arange(n_folds):
            global_variables.validated_antiviral_dt = initial_antiviral_dt     
            reset_counter() 
            model.reset_parameters()
            model.is_fine_tuning = False
            unfreeze_drug_target_parameters(model, True)    
            cv_data = create_cv_data(data, k_folds, cv_run, k_folds_dt, k_folds_ft)
            cv_data = cv_data.to(device)

            # train the model 
            losses, val_losses, masks , pot_hf_diff_cv, pot_dt_diff_cv= \
                train_model(model, optimizer, cv_data, criterion, n_epochs, drop_out_rate, drop_out_rate_edges, trial, n_fine_tuning_steps, cv_run)
            
            pot_hf_diff += pot_hf_diff_cv
            pot_dt_diff += pot_dt_diff_cv
            log_cv_masks_and_losses(cv_run, trial, cv_data, masks, losses, val_losses)
        
        if global_variables.is_hpo and trial is not None:
            trial.set_user_attr("degree_dict", global_variables.degree_dict)
            trial.set_user_attr("used_random_seed", global_variables.random_seed)
        pot_hf_diff = pot_hf_diff/n_folds
        pot_dt_diff = pot_dt_diff/n_folds
            

    else:
        model.reset_parameters()

        losses, test_losses, masks , pot_hf_diff_cv, pot_dt_diff_cv= \
            train_model(model, optimizer, data, criterion, n_epochs, drop_out_rate, drop_out_rate_edges, trial, n_fine_tuning_steps)

        log_masks_and_losses(trial, data, masks, losses, test_losses) if trial is not None else None
    
    print(f"aupr_results: {global_variables.performances.aupr_results} with mean: {np.mean(global_variables.performances.aupr_results)}")
    print(f"auroc_results: {global_variables.performances.auroc_results} with mean: {np.mean(global_variables.performances.auroc_results)}")
    print(f"aupr_results_dt: {global_variables.performances.aupr_results_dt} with mean: {np.mean(global_variables.performances.aupr_results_dt)}")
    print(f"auroc_results_dt: {global_variables.performances.auroc_results_dt} with mean: {np.mean(global_variables.performances.auroc_results_dt)}")
    print(f"best_classification_thresholds: {global_variables.performances.best_classification_thresholds} with mean: {np.mean(global_variables.performances.best_classification_thresholds)}")
    print(f"MCC_results: {global_variables.performances.MCC_results} with mean: {np.mean(global_variables.performances.MCC_results)}")
    hf_classified_as_dg_fraction = np.mean(global_variables.validation_performance.hf_classified_as_dg_fraction)
    inferred_dt_classified_as_dg_fraction = np.mean(global_variables.validation_performance.inferred_dt_classified_as_dg_fraction)
    validated_dt_classified_as_dg_fraction = np.mean(global_variables.validation_performance.validated_dt_classified_as_dg_fraction)

    print(f"Validation performance: hf_classified_as_dg_fraction = {hf_classified_as_dg_fraction} "
            f"inferred_dt_classified_as_dg_fraction = {inferred_dt_classified_as_dg_fraction} "
            f"validated_dt_classified_as_dg_fraction = {validated_dt_classified_as_dg_fraction}")



    log_performance_metrics(trial, data) if trial is not None else None
    
    if n_folds >= 2:
        result = np.mean(global_variables.performances.auroc_results)*0.7 + (np.mean(global_variables.performances.aupr_results)) - np.std(global_variables.performances.aupr_results)
        #result += np.mean(global_variables.performances.auroc_results_dt)*0.5 + np.mean(global_variables.performances.aupr_results_dt)*0.5 
        result += pot_hf_diff
        result += pot_dt_diff
    else:
        result = np.mean(global_variables.performances.auroc_results)*0.7 + (np.mean(global_variables.performances.aupr_results)) - (np.mean(test_losses)*1.8)**2 
        result += np.mean(global_variables.performances.auroc_results_dt)*0.7 + np.mean(global_variables.performances.aupr_results_dt)
    return result

def get_training_type():
    now = "_" + str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().time().hour) + "_" + str(datetime.datetime.now().time().minute) + "_" + str(datetime.datetime.now().time().second)
    training_type = ""
    #if global_variables.is_positional_encoding:
    #    if "page_rank" in global_variables.positional_encoding_file:
    #        training_type = "_page_rank"
    #    if "node2vec" in global_variables.positional_encoding_file:
    #        training_type = "_node2vec"
    #    if "personalized_page_rank" in global_variables.positional_encoding_file:
    #        training_type = "_personalized_page_rank"
    #        if global_variables.is_ppr_from_target_to_source:
    #            training_type += "_from_target_to_source"
    #        else:
    #            training_type += "_from_source_to_target"
    #if global_variables.is_positional_encoding_and_omics:
    #    training_type = "_positional_encoding_and_omics"
    
    training_type = "_" + global_variables.model_to_use
    if global_variables.is_pretraining:
        training_type += "_pretraining"
    if global_variables.is_cross_validation:
        training_type += "_cv"

    
    return training_type, now


class Performances():
    def __init__(self):
            self.aupr_results = []
            self.auroc_results = []
            self.aupr_results_dt = []
            self.auroc_results_dt = []
            self.best_classification_thresholds = []
            self.MCC_results = []

class ValidationPerformances():
    def __init__(self):
        self.hf_classified_as_dg_fraction = []
        self.inferred_dt_classified_as_dg_fraction = []
        self.validated_dt_classified_as_dg_fraction = [] 

def unfreeze_drug_target_parameters(model, value: bool):
    model.linear_drug_targetable.weight.requires_grad = value
    model.linear_drug_targetable.bias.requires_grad = value
    model.out_drug_targetable.weight.requires_grad = value
    model.out_drug_targetable.bias.requires_grad = value

def return_new_edge_mask(edge_mask, edge_index, high_prob_indices, low_prob_indices):
    row, _ = edge_index
    #edge_mask_values_to_flip = torch.rand(row.size(0), device=edge_index.device) < probability
    # extract the edge mask values to flip based on the edge_flip_distribution 

    high_prob_indices = torch.where(high_prob_indices)[0]
    low_prob_indices = torch.where(low_prob_indices)[0]
    #high_prob_indices = random.sample(high_prob_indices.tolist(), 1000)
    low_prob_indices = random.sample(low_prob_indices.tolist(), 1000) 
    edge_mask_values_to_flip = torch.zeros(row.size(0), dtype=torch.bool, device=edge_index.device)
    #edge_mask_values_to_flip[high_prob_indices] = True
    edge_mask_values_to_flip[low_prob_indices] = True
    edge_mask[edge_mask_values_to_flip] = ~edge_mask[edge_mask_values_to_flip]
    return edge_mask_values_to_flip

def save_model_with_lowest_validation_loss(model, test_loss, val_losses, early_stopping, now, epoch, n_epochs, pt_string = ""):
    val_losses.append(test_loss.item())
    if len(val_losses) > 5:
        early_stopping(np.mean(val_losses[-2:]), model)
    if early_stopping.early_stop:
        print("Early stopping")
        model.load_state_dict(torch.load(global_variables.home_folder + "GNN/best_models/" + global_variables.dataset_type + now + pt_string + "_best_model.pt"))
        epoch = n_epochs - 1
        early_stopping.reset()
    return epoch

def train_model(model, optimizer, data, criterion, n_epochs, drop_out_rate, drop_out_rate_edges, trial, n_fine_tuning_steps, cv_run=None):
    losses = []
    test_losses = []
    masks_list = []

    # generare a unique name based on the exact time
    now = "_" + str(datetime.datetime.now().date()) + "_" + str(datetime.datetime.now().time().hour) + "_" + str(datetime.datetime.now().time().minute) + "_" + str(datetime.datetime.now().time().second) + "_" + str(datetime.datetime.now().time().microsecond) + "_" + str(random.randint(1000, 9999))
    early_stopping = models.EarlyStopping(patience=1000, verbose=True, delta=-0.001, path=global_variables.home_folder + "GNN/best_models/" + global_variables.dataset_type + now + "_best_model.pt")
    early_stopping_pt = models.EarlyStopping(patience=1000, verbose=True, delta=-0.001, path=global_variables.home_folder + "GNN/best_models/" + global_variables.dataset_type + now + "_pt_best_model.pt")
    val_losses_ft = []
    val_losses_pt = []
    epoch = 0
    if not global_variables.is_pretraining:
        epoch = n_epochs - n_fine_tuning_steps if n_epochs > n_fine_tuning_steps  else n_epochs//2
    while epoch < n_epochs:

        # # use the GNN explainer in the middle of the pre-training to retrieve the most important edges
        # final_pt_steps_after_denoising = 10
        # model.is_gnn_explainer = True if epoch == (n_epochs - n_fine_tuning_steps - final_pt_steps_after_denoising if n_epochs > n_fine_tuning_steps + final_pt_steps_after_denoising else n_epochs//2 - final_pt_steps_after_denoising) else False
        # if model.is_gnn_explainer:
        #    print("Start GNN explainer and load best pre-training model")
        #    model.load_state_dict(torch.load(global_variables.home_folder + "GNN/best_models/" + global_variables.dataset_type + now + "_pt_best_model.pt"))
        #    model.is_gnn_explainer = False

        if (epoch == (n_epochs - n_fine_tuning_steps if n_epochs > n_fine_tuning_steps  else n_epochs//2)) or early_stopping_pt.early_stop:

            print("Load best pre-training model")
            model.load_state_dict(torch.load(global_variables.home_folder + "GNN/best_models/" + global_variables.dataset_type + now + "_pt_best_model.pt"))

            print("Start fine tuning")
            #start_fine_tuning(data, model)
            model.is_fine_tuning = True
            # reduce learning rate
            optimizer = torch.optim.Adam(model.parameters(), 
                                         lr=global_variables.learning_rate[0]/3,
                                            weight_decay=global_variables.decay[0])
            # reset masks list
            masks_list = []
            # freeze drug target final linear layers parameters
            unfreeze_drug_target_parameters(model, False)
            #model.message_weights.requires_grad = False
            if early_stopping_pt.early_stop:
                epoch = (n_epochs - n_fine_tuning_steps if n_epochs > n_fine_tuning_steps  else n_epochs//2)
                early_stopping_pt.reset()

        if model.is_fine_tuning:
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([global_variables.negative_weight, global_variables.positive_weight], device=data.x.device))
            loss_func = lambda out, _, data, train_mask_dg, __: criterion(out[train_mask_dg].squeeze(1), target = data.y_ft[train_mask_dg])
        else:
            criterion = torch.nn.CrossEntropyLoss()
            loss_func = lambda out, out_dt, data, train_mask_dg, train_mask_dt, dg_targets=data.y , criterion_dg=criterion: criterion_dg(out[train_mask_dg].squeeze(1), target = dg_targets[train_mask_dg]) + 1.5*criterion(out_dt[train_mask_dt].squeeze(1), target = data.y_drug_target[train_mask_dt])

        loss, test_loss = train(model, optimizer, data, loss_func, drop_out_rate, drop_out_rate_edges, masks_list)
        # save model with the lowest validation loss when pre-training

        # Call the function in the main code
        if not model.is_fine_tuning:
            epoch = save_model_with_lowest_validation_loss(model, test_loss, val_losses_pt, early_stopping_pt, now, epoch, n_epochs, "_pt")
        elif model.is_fine_tuning:
        # save model with the lowest validation loss when fine tuning
            epoch = save_model_with_lowest_validation_loss(model, test_loss, val_losses_ft, early_stopping, now, epoch, n_epochs)        

        disese_gene_positive_ratio = data.y[masks_list[-1].train_mask].sum()/masks_list[-1].train_mask.sum() if not model.is_fine_tuning else data.y_ft[masks_list[-1].train_mask].sum()/masks_list[-1].train_mask.sum()
        print(f"disease gene positive vs negative ratio {disese_gene_positive_ratio} drug target positive vs negative ratio {data.y_drug_target[masks_list[-1].train_mask_dt].sum()/masks_list[-1].train_mask_dt.sum()}")
        losses.append(loss.item())
        test_losses.append(test_loss.item())  
        train_set_metrics, test_set_metrics, train_set_metrics_dt, test_set_metrics_dt, _, pred_dt = test(model, data, masks_list[-1]) 
        
        print(f'Epoch disease genes: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_set_metrics["accuracy"]:.4f}, Test Acc: {test_set_metrics["accuracy"]:.4f}, Train auroc: {train_set_metrics["auroc"]:.4f}, Train aupr: {train_set_metrics["aupr"]:.4f}, Test auroc: {test_set_metrics["auroc"]:.4f}, Test aupr: {test_set_metrics["aupr"]:.4f}')  
        
        print(f'Epoch drug targets: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {np.around(test_loss.item(), decimals=4)}, Train Acc: {train_set_metrics_dt["accuracy"]:.4f}, Test Acc: {test_set_metrics_dt["accuracy"]:.4f}, Train auroc: {train_set_metrics_dt["auroc"]:.4f}, Train aupr: {train_set_metrics_dt["aupr"]:.4f}, Test auroc: {test_set_metrics_dt["auroc"]:.4f}, Test aupr: {test_set_metrics_dt["aupr"]:.4f}')  

        if (epoch == n_epochs - 1) or early_stopping.early_stop:
            #model.load_state_dict(torch.load(global_variables.home_folder + "GNN/best_models/" + global_variables.dataset_type + "_best_model.pt"))
            print(f"Load best model: {global_variables.home_folder + 'GNN/best_models/' + global_variables.dataset_type + now + '_best_model.pt'}")
            model.load_state_dict(torch.load(global_variables.home_folder + "GNN/best_models/" + global_variables.dataset_type + now + "_best_model.pt"))
            if not global_variables.is_cross_validation:            
                for iteration in range(0, 5):

                    train_set_metrics, test_set_metrics, train_set_metrics_dt, test_set_metrics_dt, preds, pred_dt = test(model, data, masks_list[-5:][iteration])
                    global_variables.performances.aupr_results_dt.append(test_set_metrics_dt["aupr"])
                    global_variables.performances.auroc_results_dt.append(test_set_metrics_dt["auroc"])
                    print(f'Last Epoch drug targets iteration {iteration}: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {np.around(test_loss.item(), decimals=4)}, Train Acc: {train_set_metrics_dt["accuracy"]:.4f}, Test Acc: {test_set_metrics_dt["accuracy"]:.4f}, Train auroc: {train_set_metrics_dt["auroc"]:.4f}, Train aupr: {train_set_metrics_dt["aupr"]:.4f}, Test auroc: {test_set_metrics_dt["auroc"]:.4f}, Test aupr: {test_set_metrics_dt["aupr"]:.4f}')  

                    global_variables.performances.aupr_results.append(test_set_metrics["aupr"])
                    global_variables.performances.auroc_results.append(test_set_metrics["auroc"])
                    global_variables.performances.best_classification_thresholds.append(test_set_metrics["best_classification_threshold"])
                    global_variables.performances.MCC_results.append(test_set_metrics["MCC"])

                    print(f'Last Epoch disease genes iteration {iteration}: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {np.around(test_loss.item(), decimals=4)}, Train Acc: {train_set_metrics["accuracy"]:.4f}, Test Acc: {test_set_metrics["accuracy"]:.4f}, Train auroc: {train_set_metrics["auroc"]:.4f}, Train aupr: {train_set_metrics["aupr"]:.4f}, Train MCC: {train_set_metrics["MCC"]:.4f}, Test auroc: {test_set_metrics["auroc"]:.4f}, Test aupr: {test_set_metrics["aupr"]:.4f}, Test MCC: {test_set_metrics["MCC"]:.4f}')    
                
                if trial is not None: 
                    trial.set_user_attr("preds", preds.tolist())
                    trial.set_user_attr("preds_dt", pred_dt.tolist())
                print(f"node names {data.node_names}")
                print(f"preds {preds}")
                potential_host_factors_mean_preds, antiviral_dt_mean_preds = analyze_antiviral_targets(data, preds, pred_dt, global_variables.validated_antiviral_dt, test_set_metrics_dt["best_classification_threshold"])
            
            else:
                auprs = []
                aurocs = []
                MCCs = []
                class_thresholds = []
                auprs_dt = []
                aurocs_dt = []
                for iteration in range(0, 3): #TODO use only one mask to evaluate metrics during pretraining at least!
                    train_set_metrics, val_set_metrics, train_set_metrics_dt, val_set_metrics_dt, preds, pred_dt = test(model, data, masks_list[-3:][iteration])
                    auprs_dt.append(val_set_metrics_dt["aupr"])
                    aurocs_dt.append(val_set_metrics_dt["auroc"])
                    print(f'Last Epoch drug targets iteration {iteration}: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {np.around(test_loss.item(), decimals=4)}, Train Acc: {train_set_metrics_dt["accuracy"]:.4f}, Val Acc: {val_set_metrics_dt["accuracy"]:.4f}, Train auroc: {train_set_metrics_dt["auroc"]:.4f}, Train aupr: {train_set_metrics_dt["aupr"]:.4f}, Val auroc: {val_set_metrics_dt["auroc"]:.4f}, Val aupr: {val_set_metrics_dt["aupr"]:.4f}')
                    auprs.append(val_set_metrics["aupr"])
                    aurocs.append(val_set_metrics["auroc"])
                    MCCs.append(val_set_metrics["MCC"])
                    class_thresholds.append(val_set_metrics["best_classification_threshold"])
                    print(f'Last Epoch disease genes iteration {iteration}: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {np.around(test_loss.item(), decimals=4)}, Train Acc: {train_set_metrics["accuracy"]:.4f}, Val Acc: {val_set_metrics["accuracy"]:.4f}, Train auroc: {train_set_metrics["auroc"]:.4f}, Train aupr: {train_set_metrics["aupr"]:.4f}, Train MCC: {train_set_metrics["MCC"]:.4f}, Val auroc: {val_set_metrics["auroc"]:.4f}, Val aupr: {val_set_metrics["aupr"]:.4f}, Val MCC: {val_set_metrics["MCC"]:.4f}')    
                
                assert cv_run is not None, "Cross validation run is not defined"
                if trial is not None:
                    trial.set_user_attr("preds cv: " + str(cv_run), preds.tolist())
                    trial.set_user_attr("preds_dt cv: " + str(cv_run), pred_dt.tolist())
  
                potential_host_factors_mean_preds, antiviral_dt_mean_preds = analyze_antiviral_targets(data, preds, pred_dt,global_variables.validated_antiviral_dt, class_thresholds)

                global_variables.performances.aupr_results.append(np.mean(auprs))
                global_variables.performances.auroc_results.append(np.mean(aurocs))
                global_variables.performances.aupr_results_dt.append(np.mean(auprs_dt))
                global_variables.performances.auroc_results_dt.append(np.mean(aurocs_dt))
                global_variables.performances.best_classification_thresholds.append(np.mean(class_thresholds))
                global_variables.performances.MCC_results.append(np.mean(MCCs))
            
            if early_stopping.early_stop:
                print("Early stopping")
                epoch = n_epochs - 1
        
        epoch += 1


    return losses, test_losses, masks_list[-1], potential_host_factors_mean_preds - preds.mean(), antiviral_dt_mean_preds - preds.mean()


if __name__ == '__main__':
    training_type, now_study = get_training_type()
    if global_variables.is_hpo:
        
        if not global_variables.is_inference:
            random_seed = global_variables.random_seed
            study_file_name = 'pna_predictions_' + str(random_seed) + '.db'
            starting_global_variables = global_variables.copy()
            #trial = 0
            # for trial in range(0, 10):
            #     print(f"start of Trial {trial}")
            #     # reset global variables after each k-fold cv run at the same original memory address that is used also by the other scripts
            #     global_variables.__dict__.update(starting_global_variables.__dict__)
            #     global_variables.random_seed = random_seed + trial
            #     print(f"random seed {global_variables.random_seed}")
            #     sampler = optuna.samplers.TPESampler(seed=global_variables.random_seed)
            #     #sampler = optuna.samplers.RandomSampler(seed=global_variables.random_seed)
            #     study = optuna.create_study(study_name='test', sampler=sampler, storage= 'sqlite:///' + global_variables.home_folder + 'GNN/gcn_random_searches/' + study_file_name, direction='maximize', load_if_exists=True)
            #     study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=True)
            print(f"random seed {global_variables.random_seed}")
            sampler = optuna.samplers.TPESampler(seed=global_variables.random_seed)
            study = optuna.create_study(study_name='test', sampler=sampler, storage= 'sqlite:///' + global_variables.home_folder + 'GNN/gcn_random_searches/' + study_file_name, direction='maximize', load_if_exists=True)
            study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)
            # Print the result
            best_trial = study.best_trial
            print(f"Best trial: score {best_trial.value}, params {best_trial.params}")
        else:
            global_variables.is_cross_validation = False
            #study_file_name = 'gcn_test_performance.db'
            study_file_name = generate_name_for_study_file(global_variables)
            print(f"study_file_name {study_file_name}")
            study_file_name += 'test_performance_'
            starting_seed = 50
            ending_seed = 70
            study_file_name += "seeds_" + str(starting_seed) + "_" + str(ending_seed) + ".db"
            model_name = global_variables.model_to_use[:3].lower() if not global_variables.is_mlp_used else "mlp"

            for random_seed in range(starting_seed, ending_seed + 1):
                print(f"Load best model for random seed {42}")
                # check if file exists
                assert os.path.exists(global_variables.home_folder + "GNN/gcn_random_searches/" + model_name + "_predictions_" + str(42) + ".db"), "File does not exist"
                study_file_name_to_load = model_name + "_predictions_" + str(42) + ".db"
                study = optuna.load_study(study_name='test', storage= 'sqlite:///' + global_variables.home_folder + 'GNN/gcn_random_searches/' + study_file_name_to_load)
                global_variables.best_params = study.best_params
                global_variables.random_seed = random_seed
                sampler = optuna.samplers.TPESampler(seed=random_seed)
                study = optuna.create_study(study_name='test', sampler=sampler, storage= 'sqlite:///' + global_variables.home_folder + 'GNN/gcn_random_searches/' + study_file_name, direction='maximize', load_if_exists=True)
                study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=True)



                # log the current seed 
                

    else:
        global_variables.is_tune_ppi = True
        global_variables.is_inference = False
        global_variables.is_cross_validation = False
        result = objective()
        print(f"Result {result}")