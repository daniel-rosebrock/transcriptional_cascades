# Fit functions to pseudotime ordered gene expression profiles from scRNA-Seq data using affine-invariant ensemble MCMC.

## Required input files:
   - count_matrix_fn: contains a .tsv file with a scRNA-Seq count matrix (cells in rows, genes in columns) where cells are ordered along a pseudotime trajectory (see [data/pancreas_beta_development_e_14_5.raw_count_matrix.tsv.gz](data/pancreas_beta_development_e_14_5.raw_count_matrix.tsv.gz) for an example)

As an example dataset, scRNA-Seq (10X) data from 4 embryonic mouse stages (E12.5-15.5) of pancreatic epithelial cells from [Bastidas-Ponce et al.](https://journals.biologists.com/dev/article/146/12/dev173849/19483/Comprehensive-single-cell-mRNA-profiling-reveals-a) was downloaded from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188). The dataset was subset to E14.5 cells of the endocrine beta cell lineage, and [diffusion pseudotime](https://www.nature.com/articles/nmeth.3971) was used to estimate pseudotime for cells along the trajectory with [scanpy](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0). See pancreatic_endocrine_differentiation_GSE132188.ipynb. In this notebook, the input file data/pancreas_beta_development_e_14_5.raw_count_matrix.tsv was generated.

## Run MCMC on a set of genes 
   - data/check.tsv contains a list of genes to run MCMC

In order to run the MCMC on an input file, the number of MCMC iterations needs to be specified (default is 10000) and an output directory must be specified. The MCMC is an affine-invariant ensemble MCMC implemented using the [emcee](https://arxiv.org/abs/1202.3665) package. Here is an example:  

    python run_mcmc.py --gene_fn data/check.tsv --count_matrix_fn data/pancreas_beta_development_e_14_5.raw_count_matrix.tsv --n_mcmc_iter 10000 --report_mcmc_progress True --output_dir outputs/e14_5_pancreas_beta_development

## Make gene subsets
Each gene can be run in parallel, for example on a HPC cluster. To generate input gene files which are then run in parallel, genes which are expressed in at least 1% of cells are split into N input files. Here is an example to generate 1000 input gene subset files (subsets are stored in data/pancreas_beta_development_e_14_5_gene_subsets):
    
    python make_gene_subsets.py --count_matrix_fn data/pancreas_beta_development_e_14_5.raw_count_matrix.tsv --n_subsets 1000 --min_expr 0.01 --output_dir data/pancreas_beta_development_e_14_5_gene_subsets/

## Run on compute cluster
To submit each gene subset in parallel, configure the bash script to use the correct input files, and execute jobs with cluster, for example:
    
    ls -1 $PWD/data/pancreas_beta_development_e_14_5_gene_subsets/ | xargs -I {} mxqsub -t 10h -N pancreas_beta_development_e_14_5 --stdout {}.stdout --stderr {}.stderr -w $PWD/outputs_full/pancreas_beta_development_e_14_5 bash $PWD/run_e14_5_pancreas_endocrine.sh $PWD $PWD/data/pancreas_beta_development_e_14_5_gene_subsets/{}

## Outputs
The outputs of each MCMC run are stored as pkls, which can be loaded using the mcmc_data class (in mcmc_data.py). The notebook load_mcmc_run_e14_5_pancreas_beta.ipynb shows an example of loading the MCMC runs for the subset of genes in data/check.tsv with various plotting examples. The notebook load_mcmc_run_full.ipynb highlights results across all genes in the dataset.
