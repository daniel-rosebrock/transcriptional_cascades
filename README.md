# Fit functions to pseudotime ordered gene expression profiles from scRNA-Seq data using affine-invariant ensemble MCMC.

## Required input files:
   - count_matrix_fn: contains a .tsv file with a scRNA-Seq count matrix (cells in rows, genes in columns) where cells are ordered along a pseudotime trajectory (see data/e14_5_mouse_forebrain_neuron_diff.raw_count_matrix.tsv for an example)

As an example dataset, the mouse brain atlas from [Manno et al.](https://www.nature.com/articles/s41586-021-03775-x) was downloaded from [UCSC Cell Browser](https://cells.ucsc.edu/?ds=mouse-dev-brain). The dataset was subset to forebrain dorsal cells of e14.5, and [diffusion pseuodtime](https://www.nature.com/articles/nmeth.3971) was used to estimate pseudotime for cells along the cortical NSC -> IP -> neuron route with [scanpy](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0). See mouse_brain_e14.5_preprocessing.ipynb. In this notebook, the input file data/e14_5_mouse_forebrain_neuron_diff.raw_count_matrix.tsv was generated.

## Run MCMC on a set of genes 
   - data/check.tsv contains a list of genes to run MCMC

In order to run the MCMC on an input file, the number of MCMC iterations needs to be specified (default is 10000) and an output directory must be specified. The MCMC is an affine-invariant ensemble MCMC implemented using the [emcee](https://arxiv.org/abs/1202.3665) package. Here is an example:  

    python run_mcmc.py --gene_fn data/check.tsv --count_matrix_fn data/e14_5_mouse_forebrain_neuron_diff.raw_count_matrix.tsv --n_mcmc_iter 10000 --report_mcmc_progress True --output_dir outputs/e14_5_forebrain_dorsal/

## Make gene subsets
Each gene can be run in parallel, for example on a HPC cluster. To generate input gene files which are then run in parallel, genes which are expressed in at least 1% of cells are split into 1000 input files. Here is an example to generate input gene subset files (subsets are stored in data/e14_5_mouse_forebrain_neuron_diff_gene_subsets):

    python make_gene_subsets.py --count_matrix_fn data/e14_5_mouse_forebrain_neuron_diff.raw_count_matrix.tsv --n_subsets 1000 --min_expr 0.01 --output_dir data/e14_5_mouse_forebrain_neuron_diff_gene_subsets

## Run on compute cluster
To submit each gene subset in parallel, configure the bash script to use the correct input files, and execute jobs with cluster, for example:

    ls -1 $PWD/data/e14_5_mouse_forebrain_neuron_diff_gene_subsets/ | xargs -I {} mxqsub -t 10h -N mcmc_e14_5_forebrain_dorsal --stdout {}.stdout --stderr {}.stderr -w $PWD/outputs_full/e16_5_forebrain_dorsal bash $PWD/run_e14_5_forebrain_dorsal.sh $PWD $PWD/data/e14_5_mouse_forebrain_neuron_diff_gene_subsets/{}

## Outputs
The outputs of each MCMC run are stored as pkls, which can be loaded using the mcmc_data class (in mcmc_data.py). The notebook load_mcmc_run.ipynb shows an example of loading the MCMC runs for the subset of genes in data/check.tsv with various plotting examples. The notebook load_mcmc_run_full.ipynb highlights results across all genes in the dataset.
