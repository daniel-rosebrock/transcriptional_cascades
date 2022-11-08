import pandas as pd
import scanpy as sc
import numpy as np
import pickle as pkl

import emcee
import math
from scipy.special import gamma
from scipy.special import loggamma

import os
from scipy.optimize import curve_fit

from lik_models import *
from helper_funcs import *

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

parser = ArgumentParser(description='Run MCMC fits on pseudotime ordered expression data')
parser.add_argument('--gene_fn', type=str, help='Genes to run MCMC fits', default=None)
parser.add_argument('--count_matrix_fn', type=str, help='Count Matrix ordered by psueodtime', default=None)
parser.add_argument('--n_mcmc_iter', type=int, help='Numer of MCMC iterations to run', default=10000)
parser.add_argument('--size_factor', type=float, help='Size factor to normalize count data', default=None)
parser.add_argument('--report_mcmc_progress', type=str, help='Print progress bar of MCMC run (boolean)', default='False')
parser.add_argument('--output_dir', type=str, help='Output directory', default=None)

args = parser.parse_args()

gene_fn = args.gene_fn
count_matrix_fn = args.count_matrix_fn
output_dir = args.output_dir
report_mcmc_progress = args.report_mcmc_progress
if report_mcmc_progress == 'False':
    report_mcmc_progress = False
else:
    report_mcmc_progress = True
n_mcmc_iter = args.n_mcmc_iter

print('Loading Count Matrix...')
counts_dict,counts_dict_by_cell,cells,genes_full = read_in_count_matrix_full(count_matrix_fn)

libsizes = np.array([np.sum(counts_dict_by_cell[cell]) for cell in cells])
median_counts = np.median(libsizes)
if args.size_factor is None:
    size_factor = median_counts
else:
    size_factor = args.size_factor

print('Estimating Global Dispersion Parameter...')
phi_opt = estimate_global_dispersion_param(counts_dict,nbins=5)
print('Global phi estimate: ',str(phi_opt))

with open(gene_fn,'r') as genes:
    for i,row in enumerate(genes):
        gene = row.strip("\n")
        if os.path.exists(output_dir+'/'+gene+'.pkl'):
            print('gene already run:', gene)
            continue
        print('running gene:', gene)
        #normalize expression data
        counts = np.array(counts_dict[gene])
        ordered_expression = [np.log(c/l*size_factor+1) for c,l in zip(counts,libsizes)]
        xdata = range(len(ordered_expression))
        max_expr = max(ordered_expression)

        ### run Double Sigmoidal MCMC ###
        print('running Double Sigmoidal MCMC')
        #p0 = [0.01, max(ordered_expression),0.01,np.median(xdata)/2.,np.median(xdata)+np.median(xdata)/2.,0.1,0.1] # this is n mandatory initial guess
        pos_new = []
        #Note: These parameters ensure all walkers start in a parameter setting with positive likelihood
        for b_min in np.linspace(0.01*max_expr,max_expr*0.99,9):
            pos_new.append(np.array([b_min, max_expr/2.+np.random.uniform(-max_expr/4.,max_expr/4.),
                                     0.1*max_expr+np.random.uniform(-max_expr*0.05,max_expr*0.05),
                                     np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     np.median(xdata)+np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     0.1+np.random.uniform(-0.05,0.05),0.1+np.random.uniform(-0.05,0.05)])) # this is n mandatory initial guess)
        for b_min in np.linspace(0.01*max_expr,max_expr*0.99,5):
            pos_new.append(np.array([b_min, max_expr/2.+np.random.uniform(-max_expr/4.,max_expr/4.),
                                     max_expr+np.random.uniform(-max_expr/2.,0),
                                     np.median(xdata)/2.+np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     np.median(xdata)+np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     0.1+np.random.uniform(-0.05,0.05),0.1+np.random.uniform(-0.05,0.05)])) # this is n mandatory initial guess)
        for b_max in np.linspace(0.01*max_expr,max_expr*0.99,9):
            pos_new.append(np.array([max_expr*0.1+np.random.uniform(-max_expr*0.05,max_expr*0.05),
                                     max_expr/2.+np.random.uniform(-max_expr/4.,max_expr/4.),
                                     b_max,np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     np.median(xdata)+np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     0.1+np.random.uniform(-0.05,0.05),0.1+np.random.uniform(-0.05,0.05)])) # this is n mandatory initial guess)
        for b_max in np.linspace(0.01*max_expr,max_expr*0.99,5):
            pos_new.append(np.array([max_expr+np.random.uniform(-max_expr/2.,0), 
                                     max_expr/2.+np.random.uniform(-max_expr/4.,max_expr/4.),
                                     b_max,np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     np.median(xdata)+np.median(xdata)/2.+np.random.uniform(-np.median(xdata)/10.,np.median(xdata)/10.),
                                     0.1+np.random.uniform(-0.05,0.05),0.1+np.random.uniform(-0.05,0.05)])) # this is n mandatory initial guess)
        pos = np.array(pos_new)
        
        nwalkers, ndim = pos.shape

        sampler_double_sigmoidal = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_double_sigmoidal, args=(xdata, counts, libsizes, size_factor, phi_opt, len(xdata), max_expr)
        )
        sampler_double_sigmoidal.run_mcmc(pos, n_mcmc_iter, progress=report_mcmc_progress)

        ### run Gaussian MCMC ###
        print('running Gaussian MCMC')
        pos_new = []
        p0 = [max_expr/2.,np.median(xdata),len(xdata)/20.,max_expr*0.05] # this is n mandatory initial guess
        try:
            popt_gauss, pcov = curve_fit(gaussian, xdata, ordered_expression,p0, 
                                     bounds=np.array([[0,0,0.01*len(xdata),max_expr*0.05],
                                        [max_expr,len(xdata),len(xdata)/4.,max_expr]]), 
                                     method='trf',maxfev=5000)
        except:
            popt_gauss = p0
        pos = popt_gauss + np.random.randn(16, 4)*0.1
        pos_new = []
        ## make sure p0 conforms to param constraints
        for p in pos:
            ## params for p0 are: a, x0, sigma, y_offset
            p_new = []
            for j,p_iter in enumerate(p):
                if j == 0:
                    if (p_iter < 0):
                        p_iter = np.random.uniform(0,popt_gauss[0]+0.01)
                    p_new.append(p_iter)
                elif j == 1:
                    if (p_iter < 0) or (p_iter > len(xdata)):
                        p_iter = np.random.uniform(0,len(xdata))
                    p_new.append(p_iter)
                elif j == 2:
                    p_new.append(np.abs(p_iter))
                elif j == 3:
                    if p_iter > max_expr:
                        p_iter = np.random.uniform(popt_gauss[3],max_expr)
                    p_new.append(np.abs(p_iter))
            pos_new.append(np.array(p_new))
        pos = np.array(pos_new)

        nwalkers, ndim = pos.shape

        sampler_gauss = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_gaussian, args=(xdata, counts, libsizes, size_factor, phi_opt, len(xdata), max_expr), 
        )
        sampler_gauss.run_mcmc(pos, n_mcmc_iter, progress=report_mcmc_progress)

        ### run Sigmoidal MCMC ###
        print('running Sigmoidal MCMC')
        p0 = [max_expr, np.median(xdata),0,max_expr*0.05] # this is n mandatory initial guess
        try:
            popt_sig, pcov = curve_fit(sigmoid, xdata, ordered_expression,p0, 
                                   bounds=np.array([[0,0,-0.5,0],
                                    [max_expr,len(xdata),0.5,max_expr]]), method='trf',maxfev=5000)
        except:
            popt_sig = p0
        pos = popt_sig + np.random.randn(16, 4)*0.1
        pos_new = []
        ## make sure p0 conforms to param constraints
        for p in pos:
            ## params for p0 are:  L ,x0, k, b
            p_new = []
            for j,p_iter in enumerate(p):
                if j == 0:
                    if p_iter > max_expr:
                        p_iter = np.random.uniform(popt_sig[0],max_expr)
                    p_new.append(np.abs(p_iter))
                elif j == 1:
                    if (p_iter < 0) or (p_iter > len(xdata)):
                        p_iter = np.random.uniform(0,len(xdata))
                    p_new.append(p_iter)
                elif j == 3:
                    p_new.append(np.abs(p_iter))
                else:
                    if p_iter > max_expr:
                        p_iter = np.random.uniform(popt_sig[3],max_expr)
                    p_new.append(p_iter)
            pos_new.append(np.array(p_new))
        pos = np.array(pos_new)

        nwalkers, ndim = pos.shape

        sampler_sigmoidal = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_sigmoidal, args=(xdata, counts, libsizes, size_factor, phi_opt, len(xdata), max_expr)
        )
        sampler_sigmoidal.run_mcmc(pos, n_mcmc_iter, progress=report_mcmc_progress)

        ### run Uniform MCMC ###
        print('running Unifrom MCMC')
        pos = np.array([np.array([x]) for x in np.random.uniform(0.01,max_expr,4)])
        nwalkers, ndim = pos.shape

        sampler_unif = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_uniform, args=(xdata, counts, libsizes, size_factor, phi_opt, max_expr)
        )
        sampler_unif.run_mcmc(pos, n_mcmc_iter, progress=report_mcmc_progress)

        samplers = {}
        for sampler,type_ in zip([sampler_double_sigmoidal,sampler_gauss,sampler_sigmoidal,sampler_unif],
            ['double sigmoidal','gauss','sigmoidal','uniform']):
            samplers[type_] = sampler

        info = {}
        print('Annotating useful info and getting best fit...')
        info = annotate_useful_info(samplers,info)
        #bic_subs_dict used to measure best fits based on sub-sampling to perc_subset and estimating bic on subsets
        bic_subs_dict,bic_avg_params,aic_avg_params = make_bic_estimate_subsets(samplers,info,counts,max_expr,
            libsizes,size_factor,phi_opt,xdata,perc_subset=0.98,n_subset=10000)
        info['bic_avg_params'] = bic_avg_params
        info['aic_avg_params'] = aic_avg_params
        info = measure_best_fits(samplers,info,ordered_expression,bic_subs_dict)
        info['dic_dict'] = estimate_dic(samplers,info)
        samplers['bic_subs_dict'] = bic_subs_dict

        print('Running autocorrelation analysis on best fit...')
        autocorrelation_analysis = run_autocorrelation_analysis(info,samplers)

        print('Writing pkls...')
        with open(output_dir+'/'+gene+'.pkl', 'wb') as handle:
            pkl.dump(samplers, handle, protocol=pkl.HIGHEST_PROTOCOL)

        with open(output_dir+'/'+gene+'.info.pkl', 'wb') as handle:
            pkl.dump(info, handle, protocol=pkl.HIGHEST_PROTOCOL)

        with open(output_dir+'/'+gene+'.autocorrelation_analysis.pkl', 'wb') as handle:
            pkl.dump(autocorrelation_analysis, handle, protocol=pkl.HIGHEST_PROTOCOL)
