### Folder containing the settings to perform random search with EMOGI

In the config file the hyperparameter random search ranges are set.

Launching the script 

```
emogi_random_search/make-hyband-search.sh
```

 you can sample with replacement the configurations that will be used with snakemake in the random search, this script makes use of *hyband* :

```
pip install git+https://github.com/e-dorigatti/hyperband-snakemake
```

Inside this script you can choose the name of the `<search_folder>` containing the random search configurations and the first two numbers indicate the number of configurations that will be randomly sampled (3 2 -> 2^3 sampled configurations).

The random search can be launched with the following command:

```
snakemake --snakefile <search_folder>/Snakefile
```
