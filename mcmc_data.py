from lik_models import *
from helper_funcs import *
from plotting_funcs import *

class mcmc_data():

    def __init__(self, count_matrix_fn, pkl_dir, tfs_fn, name, size_factor=None):
        self.count_matrix_fn = count_matrix_fn
        self.pkl_dir = pkl_dir
        self.name = name
        print('Loading Count Matrix...')
        counts_dict,counts_dict_by_cell,cells,genes_full = read_in_count_matrix_full(self.count_matrix_fn)
        self.counts_dict = counts_dict
        libsizes = np.array([np.sum(counts_dict_by_cell[cell]) for cell in cells])
        median_counts = np.median(libsizes)
        if size_factor is None: size_factor = median_counts
        self.expr_dict = {}
        print('Normalizing Count Matrix...')
        for gene in counts_dict:
            counts = np.array(counts_dict[gene])
            self.expr_dict[gene] = [np.log(c/l*size_factor+1) for c,l in zip(counts,libsizes)]
        self.tfs = [x.strip("\n") for x in open(tfs_fn,'r').readlines()]

    def __hash__(self):
        return self.name

    def __eq__(self, other):
        return self is other

    def load_pkls(self,sub_tfs=True):
        #load in all pkls
        if sub_tfs:
            self.mode_liks_full,self.med_liks_full,self.max_args_full, \
            self.best_fits,self.inflection_points,self.inflection_points_2, \
            self.inflection_point_derivs,self.inflection_point_derivs_2, \
            self.bic_mode_liks,self.bic_avg_params,self.aic_avg_params, \
            self.dic_dict,self.poor_fits = \
            load_in_all_pkls(self.pkl_dir,sub_genes=self.tfs)
        else:
            self.mode_liks_full,self.med_liks_full,self.max_args_full, \
            self.best_fits,self.inflection_points,self.inflection_points_2, \
            self.inflection_point_derivs,self.inflection_point_derivs_2, \
            self.bic_mode_liks,self.bic_avg_params,self.aic_avg_params, \
            self.dic_dict,self.poor_fits = \
            load_in_all_pkls(self.pkl_dir)

    def get_top_genes(self,n_top=50,sub_tf=True):
        bic_diff = {}
        for gene in self.best_fits.keys():
            if self.best_fits[gene] != 'uniform':
                #here, use bic estimate over avg params to sort genes by fit
                bic_diff[gene] = self.bic_avg_params[gene][self.best_fits[gene]]-\
                self.bic_avg_params[gene]['uniform']  
        self.sorted_mode_bic = sorted(bic_diff.items(),key=lambda x:x[1])
        self.sorted_mode_bic_tfs = [x for x in self.sorted_mode_bic if 
            x[0].upper() in self.tfs]
        if sub_tf == True:
            self.genes_good_fit = [x[0] for x in self.sorted_mode_bic_tfs[:n_top]]
        else:
            self.genes_good_fit = [x[0] for x in self.sorted_mode_bic[:n_top]]
    
    def get_gene_relationships(self,sub_tf=True):
        if sub_tf == True:
            self.gene_relationships = build_gene_relationships(self.genes_good_fit, \
                self.best_fits,self.inflection_points,self.inflection_point_derivs, \
                self.inflection_points_2,self.inflection_point_derivs_2,tfs=self.tfs)
        else: self.gene_relationships = build_gene_relationships(self.genes_good_fit, \
                self.best_fits,self.inflection_points,self.inflection_point_derivs, \
                self.inflection_points,self.inflection_point_derivs,tfs=None)