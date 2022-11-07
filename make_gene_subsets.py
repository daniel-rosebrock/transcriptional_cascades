import math
from helper_funcs import *

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

parser = ArgumentParser(description='Make gene subsets')
parser.add_argument('--count_matrix_fn', type=str, help='Count Matrix ordered by psueodtime', default=None)
parser.add_argument('--n_subsets', type=int, help='Numer of subset files to make', default=1000)
parser.add_argument('--min_expr', type=int, help='Minimum percent of cells where gene is expressed', default=0.01)
parser.add_argument('--output_dir', type=str, help='Output directory', default=None )

args = parser.parse_args()

count_matrix_fn = args.count_matrix_fn
min_expr = args.min_expr
n_subsets = args.n_subsets
output_dir = args.output_dir

print('Loading Count Matrix...')
counts_dict,counts_dict_by_cell,cells,genes_full = read_in_count_matrix_full(count_matrix_fn)

n_cells = len(counts_dict_by_cell)
perc_expr = {}
for gene in counts_dict:
    n_expr = sum(np.array(counts_dict[gene])>0)
    perc_expr[gene] = n_expr/n_cells

genes = [gene for gene in perc_expr if perc_expr[gene] >= min_expr]
print('Number of genes with perc. express > min_expr: ',str(len(genes)))

print('Making gene subset files...')
n_genes = len(genes)
added_genes = []
iter_ = int(math.ceil(n_genes/n_subsets))

for k in range(0,n_subsets):
    with open(output_dir+'/genes_sub.'+str(k)+'.tsv','w') as output:
        output.write('\n'.join(genes[iter_*k:iter_*(k+1)]))
    added_genes.extend(genes[iter_*k:iter_*(k+1)])

if iter_*(k+1) < len(genes)-1:
    with open(output_dir+'/genes_sub.'+str(k+1)+'.tsv','w') as output:
        output.write('\n'.join(genes[iter_*(k+1):]))
    added_genes.extend(genes[iter_*(k+1):])

print(n_genes,len(added_genes))
print(genes==added_genes)