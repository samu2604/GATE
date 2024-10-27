# This script loads the similarity matrix chunks computed in the previous step and aggregates them in a single matrix. The final matrix is saved in a csv file that can be read with pandas in python.

gene_names <- readLines('/vol/GhostFreePro/data_preprocessing_pipeline/artivir_data/positional_encoding/gene_names_ppr_goterms_filtered.txt')
num_genes <- length(gene_names)
similarity_matrix <- matrix(0.0, nrow=num_genes, ncol=num_genes)

indices_array <- c(1, 76, 153, 230, 308, 386, 465,   544,   624,   703, 784,   865,   946,  1027,  1110,  1192,  1275,  1359,  1443,
1528,  1613,  1699,  1785,  1872,  1959,  2047,  2136,  2225, 2314,  2405,  2496,  2587,  2680,  2773,  2867,  2961,  3056, 
3152,  3249,  3346,  3444,  3543,  3643,  3744,  3846,  3948, 4052,  4156,  4262,  4368,  4476,  4584,  4694,  4805,  4917, 
5030,  5145,  5261,  5378,  5497,  5617,  5738,  5861,  5986, 6113,  6241,  6371,  6503,  6637,  6773,  6912,  7052,  7195, 
7341,  7490,  7641,  7795,  7953,  8114,  8279,  8448,  8621, 8798,  8981,  9169,  9363,  9564,  9772,  9988, 10214, 10450, 
10698, 10960, 11239, 11539, 11865, 12226, 12635, 13121, 13754, 15282)

for (i in 1:length(indices_array)) {
  start_index <- as.integer(indices_array[i])
  end_index <- as.integer(indices_array[i+1]) - 1
  print(paste("start_index:", start_index, "end_index:", end_index))
  similarity_matrix_partial <- readRDS(paste0("/vol/artivir_data/gosemsim_matrices/go_sem_sim_matrix/similarity_matrix_avg_", start_index, "_", end_index, ".rds"))
  similarity_matrix <- similarity_matrix_partial + similarity_matrix
}
# save the similarity matrix in a csv file readable with pandas in python
write.csv(similarity_matrix, file = "/mnt/test/similarity_matrix_avg.csv", row.names = FALSE)

