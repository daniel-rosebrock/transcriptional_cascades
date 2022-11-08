### Fit functions to pseudotime ordered gene expression profiles from scRNA-Seq data using affine-invariant ensemble MCMC

## Required input files:
     - count_matrix_fn: contains a .tsv file with a scRNA-Seq count matrix (cells in rows, genes in columns) where cells are ordered along a pseudotime trajectory (see data/e14_5_mouse_forebrain_neuron_diff.raw_count_matrix.tsv for example)

## Run MCMC on a set of genes 
     - data/check.tsv contains a list of genes to run on (see data/check.tsv for example)

In order to run the MCMC on an input file, number of MCMC iterations needs to be specified (default is 10000) and an output directory. Here is an example:  

    python run_mcmc.py --gene_fn data/check.tsv --count_matrix_fn data/e14_5_mouse_forebrain_neuron_diff.raw_count_matrix.tsv --n_mcmc_iter 10000 --report_mcmc_progress True --output_dir outputs/e14_5_forebrain_dorsal/

## Make gene subsets
Each gene can be run in parallel, for example on a HPC cluster. To generate input gene files which are then run in parallel, genes which are expressed in at least 1% of cells are split into 1000 input files. Here is an example to generate input gene subset files (subsets are stored in data/e14_5_mouse_forebrain_neuron_diff_gene_subsets):

    python make_gene_subsets.py --count_matrix_fn data/e14_5_mouse_forebrain_neuron_diff.raw_count_matrix.tsv --n_subsets 1000 --min_expr 0.01 --output_dir data/e14_5_mouse_forebrain_neuron_diff_gene_subsets

## Run on compute cluster
To submit each gene subset in parallele, configure the bash script to use the correct input files, and execute jobs with cluster, for example:

    ls -1 $PWD/data/e14_5_mouse_forebrain_neuron_diff_gene_subsets/ | xargs -I {} mxqsub -t 10h -N mcmc_e14_5_forebrain_dorsal --stdout {}.stdout --stderr {}.stderr -w $PWD/outputs_full/e16_5_forebrain_dorsal bash $PWD/run_e14_5_forebrain_dorsal.sh $PWD $PWD/data/e14_5_mouse_forebrain_neuron_diff_gene_subsets/{}

## Outputs
The outputs of each MCMC run are stored as pkls, which can be loaded using the mcmc_data class (in mcmc_data.py). The notebook load_mcmc_run.ipynb shows an example of loading the MCMC runs for the subset of genes in data/check.tsv with various plotting examples. 
