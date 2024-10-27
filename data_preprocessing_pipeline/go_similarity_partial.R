# This script reads the list of genes used in the STRING PPI  network. It looks for the similarity between each pair of genes in the list in one direction, 
# filling the upper triangular part of the similarity matrix. The script is parallelized in the cluster, and the similarity matrix is saved in chunks files 

args <- commandArgs(trailingOnly = TRUE)

# Ensure we have the required arguments
if (length(args) != 2) {
  stop("You must supply start_index and end_index as arguments.")
}

#indices_array <- c(1, 599,  625,  654,  687,  727,  774,  831,  904, 1000, 1138, 1348, 1756, 4239)
#indices_array <- cumsum(indices_array)

indices_array <- c(1, 76, 153, 230, 308, 386, 465,   544,   624,   703, 784,   865,   946,  1027,  1110,  1192,  1275,  1359,  1443,
1528,  1613,  1699,  1785,  1872,  1959,  2047,  2136,  2225, 2314,  2405,  2496,  2587,  2680,  2773,  2867,  2961,  3056, 
3152,  3249,  3346,  3444,  3543,  3643,  3744,  3846,  3948, 4052,  4156,  4262,  4368,  4476,  4584,  4694,  4805,  4917, 
5030,  5145,  5261,  5378,  5497,  5617,  5738,  5861,  5986, 6113,  6241,  6371,  6503,  6637,  6773,  6912,  7052,  7195, 
7341,  7490,  7641,  7795,  7953,  8114,  8279,  8448,  8621, 8798,  8981,  9169,  9363,  9564,  9772,  9988, 10214, 10450, 
10698, 10960, 11239, 11539, 11865, 12226, 12635, 13121, 13754, 15282)

#indices_array <- c(1,2)

start_index <- as.integer(indices_array[as.integer(args[1])])
end_index <- as.integer(indices_array[as.integer(args[2])]) - 1
# print the indices
print(paste("start_index:", start_index, "end_index:", end_index))

library(GOSemSim)

hsGO2 <- godata('org.Hs.eg.db', keytype = "SYMBOL", ont="BP", computeIC=FALSE)

# load the genes from the positional encoding, filtered for genes not included in the GOSemSim package
gene_names <- readLines('/home/icb/samuele.firmani/GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/gene_names_ppr_goterms_filtered.txt') # nolint

# filter gene names that are not recognized in the GO database
num_genes <- length(gene_names)


# Compute pairwise similarities for this chunk
similarity_matrix_partial <- matrix(0.0, nrow=num_genes, ncol=num_genes)
for (i in start_index:end_index) {
  # print current time
  print(Sys.time())
  for (j in (i+1):num_genes) {
    print(paste("i:", i, "j:", j))
    similarity_matrix_partial[i, j] <- geneSim(gene_names[i], gene_names[j], semData=hsGO2, measure = "Wang", combine = "avg")$geneSim # nolint
    similarity_matrix_partial[j, i] <- similarity_matrix_partial[i, j] # symmetric matrix
  }
}

saveRDS(similarity_matrix_partial, file=paste0("/lustre/groups/crna01/workspace/samuele/gosemsim_matrices/go_sem_sim_matrix/similarity_matrix_avg_", start_index, "_", end_index, ".rds")) # nolint
