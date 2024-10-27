import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
import numpy as np
from sklearn.model_selection import StratifiedKFold
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing_pipeline.feature_preselection_transcriptome_proteome import global_variables
def check_updated_weights(model, previous_params):
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Calculate the absolute change in parameter
            param_change = torch.norm(previous_params[name] - param.detach(), p=2).item()
            #if param_change == 0:
            #    print(f"Parameter {name} has not changed.")
            #else:
            #    print(f"Parameter {name} has changed by {param_change:.6f}.")
            # Update the previous parameter value
            previous_params[name] = param.clone().detach()


def generate_name_for_study_file(global_variables):
    name = global_variables.fine_tuning_virus + "_"
    name += "mlp_" + str(global_variables.is_mlp_used) + "_"
    if not global_variables.is_mlp_used:
        name += "model_" + global_variables.model_to_use + "_"
    name += "pe_" + str(global_variables.is_pe_used) + "_"
    name += "prot_emb_" + str(global_variables.is_prot_emb_used) + "_"
    name += "gosemsim_" + str(global_variables.is_gosemsim_used) + "_"
    name += "omics_" + str(global_variables.is_omics_used) + "_"
    return name

def check_if_drug_targets_are_classified_as_disease_genes(data, pred, validated_antiviral_dt, counter_max=6, is_used_fixed_list=False):
    if not hasattr(check_if_drug_targets_are_classified_as_disease_genes, 'counter'):
        check_if_drug_targets_are_classified_as_disease_genes.counter = 0  # Initialize if not already done

    if is_used_fixed_list and check_if_drug_targets_are_classified_as_disease_genes.counter < counter_max:
        print("Using fixed list of weak genes to enhance the positive class for antiviral drug targets")
        #list_of_weak_genes = ["chrm1", "dhfr", "gabrr2", "htr1e", "kcnh6", "noxo1", "scn11a", "slc6a1", "oprd1", "gsto2"]
        #list_of_weak_genes = ["GSTP1","KCNH2","PPARA","NR1I3","MTTP","POMC","NTRK2","F2","THRA","DHFR","NTRK1","ESR2","ITGB3","OPRM1"]
        list_of_weak_genes = ["GSTP1","KCNH2","PPARA","NR1I3","MTTP","POMC","F2","THRA","DHFR","NTRK1"]#,"ESR2","ITGB3","OPRM1"]
        list_of_weak_genes = [x.lower() for x in list_of_weak_genes]
        common_genes_indices = np.where(np.isin(data.node_names, list_of_weak_genes))[0]
        # remove common genes from validated_antiviral_dt and return it
        validated_antiviral_dt = np.setdiff1d(validated_antiviral_dt, list_of_weak_genes)
        # if len common_genes_indices is smaller than counter_max throw an error
        if len(common_genes_indices) < counter_max:
            raise ValueError(f"Number of weak genes is {len(common_genes_indices)} which is smaller than counter_max {counter_max}")
    elif check_if_drug_targets_are_classified_as_disease_genes.counter < counter_max:
        predicted_disease_genes = data.node_names[pred > 0.75]
        # find gene in validated_antiviral_dt contained in predicted_disease_genes
        common_genes = np.intersect1d(validated_antiviral_dt, predicted_disease_genes)
        common_genes_indices = np.where(np.isin(data.node_names, common_genes))[0]
        if len(common_genes_indices) > counter_max - check_if_drug_targets_are_classified_as_disease_genes.counter:
            # select counter_max random indices from common_genes_indices
            common_genes_indices = np.random.choice(common_genes_indices, counter_max, replace=False)
            common_genes = data.node_names[common_genes_indices]
        # remove common genes from validated_antiviral_dt and return it
        validated_antiviral_dt = np.setdiff1d(validated_antiviral_dt, common_genes)
    else:
        return validated_antiviral_dt, data
    # set the labels for the common genes to 1
    for index in common_genes_indices:
        if data.y_ft[index] == 0 and check_if_drug_targets_are_classified_as_disease_genes.counter < counter_max:
            data.y_ft[index] = int(1)
            print(f"found new antiviral dt {data.node_names[index]} with counter {check_if_drug_targets_are_classified_as_disease_genes.counter}")
            # update the training mask for the new antiviral drug target
            data.train_mask_ft[index] = True
            check_if_drug_targets_are_classified_as_disease_genes.counter += 1
    
    return validated_antiviral_dt, data

# Reset the counter
def reset_counter():
    check_if_drug_targets_are_classified_as_disease_genes.counter = 0


def analyze_antiviral_targets(data, preds, pred_dt, validated_antiviral_dt, class_thresholds):
    inferred_antiviral_dt = ['mapk11', 'treml4', 'smad1', 'fos', 'tbk1', 'jun', 'arl6ip6', 'tbk1', 'rela', 'casp7', 'ikbke', 'nfkb1', 'lnpep', 'irf3',
                    'stat2', 'atf4', 'irf9', 'fadd', 'atf6', 'bach1', 'akt1', 'tbp','alg5', 'tp53', 'tcf12', 'agtr1', 'agtr2', 'stat3', 'egfr','keap1', 
                    'jund', 'cul3', 'ahr', 'fosl1', 'mas1', 'rbx1', 'tlr9']
    
    # load potential host factors
    if global_variables.fine_tuning_virus == "SARS-CoV-2":
        host_factors_path =  os.getcwd() + "/data_preprocessing_pipeline/artivir_data/host_factors_from_publications.xlsx" 
        host_factors = pd.ExcelFile(host_factors_path)
        host_factors_dfs = {sheet_name: host_factors.parse(sheet_name) for sheet_name in host_factors.sheet_names}
        potential_host_factors_list = host_factors_dfs["host_factors"]["Gene name"].str.lower().unique()
        potential_host_factors_list = np.array([host_factor.lower() for host_factor in potential_host_factors_list])

        host_factors_df = host_factors_dfs["host_factors"]
        functionally_validated_hfs = host_factors_df[host_factors_df["Functionally validated by authors"] == 'Yes']["Gene name"].unique()
    elif global_variables.fine_tuning_virus == "MPXV":
        potential_host_factors_list = global_variables.potential_positive_genes_mpxv
        functionally_validated_hfs = global_variables.host_factors_mpxv
    
    #filter potential host factors that are alsofunctionally validated host factors
    potential_host_factors_list = [hf for hf in potential_host_factors_list if hf not in functionally_validated_hfs]

    potential_host_factors_node_names_mask = np.isin(data.node_names, potential_host_factors_list)
    potential_host_factors_mean_preds = preds[potential_host_factors_node_names_mask].mean()
    fraction_of_potential_host_factors_above_dg_threshold = (preds[potential_host_factors_node_names_mask] > np.mean(class_thresholds)).sum() / len(potential_host_factors_list)
    global_variables.validation_performance.hf_classified_as_dg_fraction.append(fraction_of_potential_host_factors_above_dg_threshold.cpu().item())
    potential_host_factors_mean_preds_dt = pred_dt[potential_host_factors_node_names_mask].mean()

    # print the fractin of genes in general classified as disease genes
    fraction_of_genes_above_dg_threshold = (preds > np.mean(class_thresholds)).sum() / len(data.node_names)
    print(f"fraction of genes above dg threshold {fraction_of_genes_above_dg_threshold}")

    # filter antiviral dt that are functionally validated host factors
    inferred_antiviral_dt = [x.lower() for x in inferred_antiviral_dt if (x.lower() in data.node_names) and (x.lower() not in functionally_validated_hfs)]
    inferred_antiviral_dt_node_names_mask = np.isin(data.node_names, inferred_antiviral_dt)
    inferred_antiviral_dt_mean_preds = preds[inferred_antiviral_dt_node_names_mask].mean()
    fraction_of_inferred_antiviral_dt_above_dg_threshold = (preds[inferred_antiviral_dt_node_names_mask] > np.mean(class_thresholds)).sum() / len(inferred_antiviral_dt)
    global_variables.validation_performance.inferred_dt_classified_as_dg_fraction.append(fraction_of_inferred_antiviral_dt_above_dg_threshold.cpu().item())
    inferred_antiviral_dt_mean_preds_dt = pred_dt[inferred_antiviral_dt_node_names_mask].mean()

    validated_antiviral_dt = [x.lower() for x in validated_antiviral_dt if (x.lower() in data.node_names) and (x.lower() not in functionally_validated_hfs)]
    validated_antiviral_dt_node_names_mask = np.isin(data.node_names, validated_antiviral_dt)
    validated_antiviral_dt_mean_preds = preds[validated_antiviral_dt_node_names_mask].mean()
    fraction_of_validated_antiviral_dt_above_dg_threshold = (preds[validated_antiviral_dt_node_names_mask] > np.mean(class_thresholds)).sum() / len(validated_antiviral_dt)
    global_variables.validation_performance.validated_dt_classified_as_dg_fraction.append(fraction_of_validated_antiviral_dt_above_dg_threshold.cpu().item())
    validated_antiviral_dt_mean_preds_dt = pred_dt[validated_antiviral_dt_node_names_mask].mean()

    print(f"potential host factors mean preds as disease genes {potential_host_factors_mean_preds} while the mean max and min are {preds.mean()}, {preds.max()}, {preds.min()} and fraction of potential host factors above dg threshold {fraction_of_potential_host_factors_above_dg_threshold}")
    print(f"mean preds as drug targets {potential_host_factors_mean_preds_dt} while the mean, max and min are {pred_dt.mean()}, {pred_dt.max()}, {pred_dt.min()}")
    print(f"antiviral drug targets mean preds as disease genes {inferred_antiviral_dt_mean_preds} while the mean max and min are {preds.mean()}, {preds.max()}, {preds.min()} and fraction of inferred antiviral dt above dg threshold {fraction_of_inferred_antiviral_dt_above_dg_threshold}")
    print(f"mean preds as drug targets {inferred_antiviral_dt_mean_preds_dt} while the mean, max and min are {pred_dt.mean()}, {pred_dt.max()}, {pred_dt.min()}")
    print(f"validated antiviral drug targets mean preds as disease genes {validated_antiviral_dt_mean_preds} while the mean max and min are {preds.mean()}, {preds.max()}, {preds.min()} and fraction of validated antiviral dt above dg threshold {fraction_of_validated_antiviral_dt_above_dg_threshold}")
    print(f"mean preds as drug targets {validated_antiviral_dt_mean_preds_dt} while the mean, max and min are {pred_dt.mean()}, {pred_dt.max()}, {pred_dt.min()}")

    return potential_host_factors_mean_preds, validated_antiviral_dt_mean_preds #inferred_antiviral_dt_mean_preds

# create a data class or struct to store all the temporary masks and data
class Masks():
    def __init__(self, train_mask, test_mask, train_mask_dt=None, test_mask_dt=None):
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.train_mask_dt = train_mask_dt
        self.test_mask_dt = test_mask_dt

def run_positional_encoding_dimensionality_reduction(ranks, n_components = 128, is_standardized=True):
    #standardize ranks = (ranks - ranks.mean(axis = 0)) / ranks.std(axis = 0)
    if is_standardized:
        ranks = StandardScaler().fit_transform(ranks)
    print(f"Run PCA on ranks with shape {ranks.shape} and n_components {n_components}")
    pca = PCA(n_components=n_components)
    pca.fit(ranks)
    return torch.Tensor(pca.transform(ranks))


def accuracy(pred, target):
    r"""Computes the accuracy of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """

    if target.numel() == 0:
        print("Target is empty")
        return 0
    return (pred == target).sum().item() / target.numel()



def true_positive(pred, target, positive_class = 1):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        positive_class (int): The positive class value.
    """

    return int(((pred == positive_class) & (target == positive_class)).sum())



def true_negative(pred, target, negative_class = 0):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        negative_class (int): The negative class.
    """
    return int(((pred == negative_class) & (target == negative_class)).sum())



def false_positive(pred, target, positive_class):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        positive_class (int): The positive class value.
    """
    
    return int(((pred == positive_class) & (target != positive_class)).sum())



def false_negative(pred, target, negative_class):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        negative_class (int): The positive class value.
    """
    return int(((pred == negative_class) & (target != negative_class)).sum())




def precision(pred, target, positive_class):
    r"""Computes the precision
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        positive_class (int): The positive class value.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, positive_class)
    fp = false_positive(pred, target, positive_class)

    if np.isclose(tp + fp, 0):
        print("TP and FP are close to 0")
        return 0
    
    out = tp / (tp + fp)
    return out



def recall(pred, target, classes = {"positive" : 1, "negative" : 0}):
    r"""Computes the recall
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        classes (dict): The classes values.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, classes["positive"])
    fn = false_negative(pred, target, classes["negative"])

    if np.isclose(tp + fn, 0):
        print("TP and FN are close to 0")
        return 0

    out = tp / (tp + fn)
    if np.isnan(out):
        out = 0

    return out



def f1_score(pred, target, classes = {"positive" : 1, "negative" : 0}):
    r"""Computes the :math:`F_1` score
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}` of predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        classes (dict): The classes values.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, classes["positive"])
    rec = recall(pred, target, classes)

    if np.isclose(prec + rec, 0):
        print("Precision and recall are close to 0")
        return 0

    score = 2 * (prec * rec) / (prec + rec)

    return score

def k_fold(mask, y, n_folds):
        
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1234)
    label_idx = np.where(mask == 1)[0] # get indices of labeled genes
    splits = skf.split(label_idx, y[label_idx])   
    
    k_sets = []
    for train, test in splits:
        train_idx = label_idx[train]
        test_idx = label_idx[test]
        train_mask = np.zeros_like(mask)
        train_mask[train_idx] = 1
        test_mask = np.zeros_like(mask)
        test_mask[test_idx] = 1
        k_sets.append((train_mask, test_mask))
         
    return k_sets

def load_hdf_data(path, network_name='network', feature_name='features'):
    """Load a GCN input HDF5 container and return its content.

    This funtion reads an already preprocessed data set containing all the
    data needed for training a GCN model in a medical application.
    It extracts a network, features for all of the nodes, the names of the
    nodes (genes) and training, testing and validation splits.

    Parameters:
    ---------
    path:               Path to the container
    network_name:       Sometimes, there might be different networks in the
                        same HDF5 container. This name specifies one of those.
                        Default is: 'network'
    feature_name:       The name of the features of the nodes. Default is: 'features'

    Returns:
    A tuple with all of the data in the order: network, features, y_train, y_val,
    y_test, train_mask, val_mask, test_mask, node names.
    """
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names


def log_performance_metrics(trial, data):
    trial.set_user_attr("aupr_results", global_variables.performances.aupr_results)
    trial.set_user_attr("auroc_results", global_variables.performances.auroc_results)
    trial.set_user_attr("aupr_results_dt", global_variables.performances.aupr_results_dt)
    trial.set_user_attr("auroc_results_dt", global_variables.performances.auroc_results_dt)
    trial.set_user_attr("mcc_results", global_variables.performances.MCC_results)
    trial.set_user_attr("positive_dg", data.y.tolist())
    trial.set_user_attr("positive_host_factors", data.y_ft.tolist())
    trial.set_user_attr("positive_dt", data.y_drug_target.tolist())

def log_cv_masks_and_losses(cv_run, trial, cv_data, masks, losses, val_losses):
    if global_variables.is_hpo:
        trial.set_user_attr("train_mask dg_" + str(cv_run), cv_data.train_mask.tolist())
        trial.set_user_attr("test_mask dg_" + str(cv_run), cv_data.test_mask.tolist())
        trial.set_user_attr("train_mask dg_ft_" + str(cv_run), masks.train_mask.tolist())
        trial.set_user_attr("test_mask dg_ft_" + str(cv_run), masks.test_mask.tolist())
        trial.set_user_attr("train_loss_" + str(cv_run), losses)
        trial.set_user_attr("val_loss_" + str(cv_run), val_losses)

def log_masks_and_losses(trial, data, masks, losses, test_losses):
    if global_variables.is_hpo:        
        trial.set_user_attr("train_mask dg", data.train_mask.tolist())
        trial.set_user_attr("test_mask dg", data.test_mask.tolist())        
        trial.set_user_attr("train_loss", losses)
        trial.set_user_attr("val_loss", test_losses)   
        trial.set_user_attr("test_mask_ft", masks.test_mask.tolist()) 
        trial.set_user_attr("train_mask_ft", masks.train_mask.tolist()) 
        trial.set_user_attr("test_mask_dt", masks.test_mask_dt.tolist())
        trial.set_user_attr("train_mask_dt", masks.train_mask_dt.tolist())