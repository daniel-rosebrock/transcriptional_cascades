import pickle as pkl
import emcee
import numpy as np
import sys
import os

from lik_models import *
from helper_funcs import *

gene_fn = sys.argv[1]
pkl_dir = sys.argv[2]

with open(gene_fn,'r') as genes:
    for i,row in enumerate(genes):
        gene = row.strip("\n")
        print(gene)
        samplers = pkl.load(open(pkl_dir+gene+'.pkl','rb'))
        info = pkl.load(open(pkl_dir+gene+'.info.pkl','rb'))
        type_ = info['best_fit']
        sampler = samplers[type_]

        if type_ == 'double sigmoidal':
            n_param = 7
        elif type_ in ['gauss','sigmoidal']:
            n_param = 4
        else:
            n_param = 1

        autocorrelation_analysis = {}
        acf_params_tot = []
        for param in range(n_param):
            acfs = []
            for dt in range(1000):
                acfs_iter = []
                for samp_iter in range(n_param*2):
                    sample = samplers[type_].get_chain(discard=5000, flat=False)[:,info['max_args_full'][type_]][:,samp_iter,:][:,param]
                    acfs_iter.append(np.corrcoef(sample[:5000-dt],sample[dt:],rowvar=False)[0][1])
                acfs.append(acfs_iter)
            acf_params_tot.append(acfs)

        auto_corr_length_int = []
        for acfs in acf_params_tot:
            auto_corr_length = []
            for dx in range(1,1000):
                auto_corr_length.append(1+2*np.sum(acfs[:dx],axis=0))
            auto_corr_length_int.append(auto_corr_length)

        autocorr_time_estimates = []
        for j,auto_corr in enumerate(auto_corr_length_int):
            autocorr_time_estimates.append(max(np.mean(auto_corr,axis=1)))

        autocorrelation_analysis['acf_params_tot'] = acf_params_tot
        autocorrelation_analysis['auto_corr_length_int'] = auto_corr_length_int
        autocorrelation_analysis['autocorr_time_estimates'] = autocorr_time_estimates

        with open(pkl_dir+'/'+gene+'.autocorrelation_analysis.pkl', 'wb') as handle:
            pkl.dump(autocorrelation_analysis, handle, protocol=pkl.HIGHEST_PROTOCOL)
