#!/bin/bash

cwd=$1
input_fn=$2
count_matrix_fn=${cwd}/data/pancreas_beta_development_e_14_5.raw_count_matrix.tsv
n_mcmc_iter=10000
output_dir=${cwd}/outputs_full/pancreas_beta_development_e_14_5/

source /project/elkabetz_lab/Daniel/willow/bin/activate

python ${cwd}/run_mcmc.py --gene_fn ${input_fn} --count_matrix_fn ${count_matrix_fn} --n_mcmc_iter ${n_mcmc_iter} --report_mcmc_progress False --output_dir ${output_dir}
