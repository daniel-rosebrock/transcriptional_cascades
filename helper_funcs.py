import numpy as np
from lik_models import *
import os
import functools
import operator
import pickle as pkl
import itertools
from scipy import stats
from scipy.optimize import curve_fit

def read_in_expr_matrix(fn):
    expr_dict = {}
    genes = []
    with open(fn,'r') as expr_mat_fn:
        for i,row in enumerate(expr_mat_fn):
            spl = row.strip("\n").split("\t")
            if i == 0:
                for j in range(1,len(spl)):
                    expr_dict[spl[j]] = []
                    genes.append(spl[j])
            else:
                for j in range(1,len(spl)):
                    expr_dict[genes[j-1]].append(float(spl[j]))
    return expr_dict

def read_in_count_matrix(fn):
    counts_dict = {}
    genes = []
    with open(fn,'r') as expr_mat_fn:
        for i,row in enumerate(expr_mat_fn):
            spl = row.strip("\n").split("\t")
            if i == 0:
                for j in range(1,len(spl)):
                    counts_dict[spl[j]] = []
                    genes.append(spl[j])
            else:
                for j in range(1,len(spl)):
                    counts_dict[genes[j-1]].append(int(spl[j]))
    return counts_dict

def read_in_count_matrix_full(count_matrix_fn):
    counts_dict = {}
    counts_dict_by_cell = {}
    cells = []
    genes_full = []
    with open(count_matrix_fn,'r') as expr_mat_fn:
        for i,row in enumerate(expr_mat_fn):
            spl = row.strip("\n").split("\t")
            if i == 0:
                for j in range(1,len(spl)):
                    counts_dict[spl[j]] = []
                    genes_full.append(spl[j])
            else:
                cells.append(spl[0])
                counts_dict_by_cell[cells[-1]] = []
                for j in range(1,len(spl)):
                    counts_dict[genes_full[j-1]].append(int(spl[j]))
                    counts_dict_by_cell[spl[0]].append(int(spl[j]))
    return counts_dict,counts_dict_by_cell,cells,genes_full

def read_in_libsizes(fn):    
    libsizes = []
    with open(fn,'r') as expr_mat_fn:
        for i,row in enumerate(expr_mat_fn):
            spl = row.strip("\n").split("\t")
            if i == 0:
                pass
            else:
                libsizes.append(float(spl[1]))
    libsizes = np.array(libsizes)
    return libsizes

def get_binned_fits(log_x,log_y,nbins=5):
    x_bins = np.linspace(min(log_x),max(log_x),nbins)
    binned_fits = {}
    for j in range(nbins-1):
        x_dat_sub = log_x[(log_x>=x_bins[j])&(log_x<x_bins[j+1])]
        y_dat_sub = log_y[(log_x>=x_bins[j])&(log_x<x_bins[j+1])]
        slope, intercept, r, p, std_err = stats.linregress(x_dat_sub,y_dat_sub)
        binned_fits[(x_bins[j],x_bins[j+1])] = slope, intercept, r, p, std_err
    return binned_fits

def var_func_nb(x, b):
    y = x+x**2/b
    return (y)

def estimate_global_dispersion_param(counts_dict,nbins=5):
    means = np.array([np.mean(counts_dict[gene]) for gene in counts_dict.keys()])
    vars_ = np.array([np.var(counts_dict[gene]) for gene in counts_dict.keys()])
    xdata = means[means > 0]
    ydata = vars_[means > 0]
    log_x = np.log(xdata)/np.log(10)
    log_y = np.log(ydata)/np.log(10)
    #we do not want to use genes which highly deviate from the curve for fitting
    #therefore, bin data into 5 bins, and estimate dispersions after subtracting corresponding means
    #Subtract dispersion estimates for each gene from piece-wise linear regression fit
    binned_fits = get_binned_fits(log_x,log_y,nbins=nbins)
    y_diffs = []
    x_bins = []
    for x,y in zip(log_x,log_y):
        found = False
        for j,bfit in enumerate(binned_fits):
            if (x >= bfit[0]) and (x <= bfit[1]):
                y_diffs.append(y - (binned_fits[bfit][0]*x+binned_fits[bfit][1]))
                x_bins.append(j)
                found = True
                break
    y_diffs = np.array(y_diffs)
    x_bins = np.array(x_bins)
    #Within each bin, remove the genes with the top 20% of highly variable compared to piece linear fit
    highly_disp = []
    non_highly_disp = []
    for j,bfit in enumerate(binned_fits):
        y_diffs_sub = y_diffs[x_bins==j]
        orig_idx = np.where(x_bins==j)[0]
        sorted_y_diffs_sub = sorted(y_diffs_sub,reverse=True)
        top_cutoff = sorted_y_diffs_sub[int(len(sorted_y_diffs_sub)*0.2)]
        highly_disp.extend(orig_idx[y_diffs_sub>top_cutoff])
        non_highly_disp.extend(orig_idx[y_diffs_sub<=top_cutoff])
    highly_disp = np.array(highly_disp)
    non_highly_disp = np.array(non_highly_disp)
    p0 = [1] # this is n mandatory initial guess
    popt, pcov = curve_fit(var_func_nb, xdata[non_highly_disp], ydata[non_highly_disp],p0, method='lm',)
    return popt[0]

def get_mode(vals,n_bins=500):
    counts,bins = np.histogram(vals,bins=n_bins)
    return bins[np.argmax(counts)]

def histogram_intersection(vals, n_bins=100):
    flat_vals = functools.reduce(operator.iconcat, vals, [])
    max_vals = max(flat_vals)
    min_vals = min(flat_vals)
    n_val = len(vals[0])
    overlap_counts = [n_val]*n_bins
    for val in vals:
        counts,bins = np.histogram(val,range=(min_vals,max_vals),bins=n_bins)
        for j,count in enumerate(counts):
            overlap_counts[j] = min(overlap_counts[j],count)
    return (sum(overlap_counts)/n_val)

def measure_a_before_b(vals, n_bins=100):
    flat_vals = functools.reduce(operator.iconcat, vals, [])
    max_vals = max(flat_vals)
    min_vals = min(flat_vals)
    counts1,bins1 = np.histogram(vals[0],range=(min_vals,max_vals),bins=n_bins)
    counts2,bins2 = np.histogram(vals[1],range=(min_vals,max_vals),bins=n_bins)
    n_tot1 = np.sum(counts1)
    n_tot2 = np.sum(counts2)
    perc_less_or_equal = 0
    for j,count in enumerate(counts2):
        perc_less_or_equal += counts2[j]/n_tot2*np.sum(counts1[:j])/n_tot1
    return(perc_less_or_equal)

def double_sig_deriv(x,b_min, b_mid, b_max, x1, x2, k1, k2):
    return k1*(b_mid-b_min)*np.exp(-k1*(x-x1))/((1+np.exp(-k1*(x-x1)))**2)+k2*(b_max-b_mid)*np.exp(-k2*(x-x2))/((1+np.exp(-k2*(x-x2)))**2) 

def double_sig_triple_deriv(x,b_min, b_mid, b_max, x1, x2, k1, k2):
    return k1**3*(b_mid-b_min)*np.exp(-k1*(x-x1))*(1-4*np.exp(-k1*(x-x1))+np.exp(-2*k1*(x-x1)))/((1+np.exp(-k1*(x-x1)))**4)+\
    k2**3*(b_max-b_mid)*np.exp(-k2*(x-x2))*(1-4*np.exp(-k2*(x-x2))+np.exp(-2*k2*(x-x2)))/((1+np.exp(-k2*(x-x2)))**4)

def sig_deriv(x,L,x0,k,b):
    return k*L*np.exp(-k*(x-x0))/((1+np.exp(-k*(x-x0)))**2)

def sig_double_deriv(x,L,x0,k,b):
    return k**2*L*np.exp(-k*(x-x0))*(np.exp(-k*(x-x0))-1)/((1+np.exp(-k*(x-x0)))**3)

def sig_triple_deriv(x,L,x0,k,b):
    return k**3*L*np.exp(-k*(x-x0))*(1-4*np.exp(-k*(x-x0))+np.exp(-2*k*(x-x0)))/((1+np.exp(-k*(x-x0)))**4)

def gauss_deriv(x,a,x0,sigma,y_offset):
    return a*np.exp((-(x-x0)**2)/(2*sigma**2))*(-(x-x0)/sigma**2)

def gauss_double_deriv(x,a,x0,sigma,y_offset):
    return a*np.exp((-(x-x0)**2)/(2*sigma**2))*((x**2-2*x*x0+x0**2-sigma**2)/sigma**4)

def gauss_triple_deriv(x,a,x0,sigma,y_offset):
    return a*np.exp((-(x-x0)**2)/(2*sigma**2))*((-x**3+3*x**2*x0-3*x*x0**2+3*x*sigma**2-3*x0*sigma**2+x0**3)/sigma**6)

def calc_deriv_man(x,f):
    deriv = []
    for j in range(len(f)-1):
        deriv.append((f[j+1]-f[j])/(x[j+1]-x[j]))
    return deriv

def load_in_all_pkls(pkl_dir,sub_genes=None):
    mode_liks_full,med_liks_full,max_args_full = {},{},{}
    #samplers_full = {} #Note: storing samplers takes up too much memory
    best_fits,bic_mode_liks,bic_avg_params,aic_avg_params,dic_dict,poor_fits = {},{},{},{},{},{}
    inflection_points,inflection_points_2,inflection_point_derivs,inflection_point_derivs_2 = {},{},{},{}
    iter_ = 0
    print('Loading pkls...')
    for j,pkl_fn in enumerate(os.listdir(pkl_dir)):
        if '.info.pkl' not in pkl_fn: continue
        iter_+=1
        if iter_%100 == 0: print(iter_)
        gene = '.'.join(pkl_fn.split(".")[:-2])
        if sub_genes is not None:
            if gene.upper() not in sub_genes: continue
        info = pkl.load(open(pkl_dir+pkl_fn,'rb'))
        mode_liks_full[gene] = info['mode_liks_full']
        med_liks_full[gene] = info['med_liks_full']
        max_args_full[gene] = info['max_args_full']
        best_fits[gene] = info['best_fit']
        inflection_points[gene] = info['inflection_points']
        inflection_points_2[gene] = info['inflection_points_2']
        inflection_point_derivs[gene] = info['inflection_point_derivs']
        inflection_point_derivs_2[gene] = info['inflection_point_derivs_2']
        bic_mode_liks[gene] = info['bic_mode_liks']
        bic_avg_params[gene] = info['bic_avg_params']
        aic_avg_params[gene] = info['aic_avg_params']
        dic_dict[gene] = info['dic_dict']
        poor_fits[gene] = info['poor_fits']
    return mode_liks_full,med_liks_full,max_args_full,best_fits,inflection_points,inflection_points_2, \
        inflection_point_derivs,inflection_point_derivs_2,bic_mode_liks,bic_avg_params,aic_avg_params,dic_dict,poor_fits

def estimate_bic(k,n,ll):
    #bic - Bayesian Information Criteria
    return k*np.log(n)-2*ll*np.log(2)

def estimate_dic(samplers,info):
    #dic - Deviance Information Criteria
    dic_dict = {}
    for type_ in ['gauss','sigmoidal','double sigmoidal','uniform']:
        liks = np.ndarray.flatten(samplers[type_].get_log_prob(discard=5000)[:,info['max_args_full'][type_]])*np.log(2)
        dic_dict[type_] = -2*np.mean(liks)+2*np.var(liks)
    return dic_dict

def estimate_aic(k,ll):
    #bic - Bayesian Information Criteria
    return k*2-2*ll*np.log(2)

def make_bic_estimate_subsets(samplers,info,counts,max_expr,libsizes,
    median_counts,phi_opt,xdata,perc_subset=0.98,n_subset=10000):
    xdata = np.array(xdata)
    #len_x = len(xdata)
    len_x = xdata[-1]
    bic_avg_params = {}
    aic_avg_params = {}
    bic_subs_dict = {'double sigmoidal':[],'gauss':[],'sigmoidal':[],'uniform':[]}
    theta_dict = {} #theta_dict stores mean param estimates across mcmc runs
    for type_ in ['double sigmoidal','sigmoidal','gauss','uniform']:
        params = samplers[type_].get_chain(discard=5000,flat=False)[:,info['max_args_full'][type_]]
        params = np.concatenate(params,axis=0)
        #Note: Here we use mean to estimate optimal parameters instead of mode, potentially change to mode...
        theta_dict[type_] = np.mean(params,axis=0)
    
    #Estimate BIC using avg param estimates
    lp_double_sig = log_probability_double_sigmoidal(theta_dict['double sigmoidal'], xdata, 
                                     np.array(counts),np.array(libsizes), median_counts, phi_opt, len_x, max_expr)
    lp_sig = log_probability_sigmoidal(theta_dict['sigmoidal'], xdata, 
                                     np.array(counts),np.array(libsizes), median_counts, phi_opt, len_x, max_expr)
    lp_gauss = log_probability_gaussian(theta_dict['gauss'], xdata, 
                                     np.array(counts),np.array(libsizes), median_counts, phi_opt, len_x, max_expr)
    lp_unif = log_probability_uniform(theta_dict['uniform'], xdata, 
                                     np.array(counts),np.array(libsizes), median_counts, phi_opt, max_expr)
    
    bic_avg_params['double sigmoidal'] = estimate_bic(7,len_x,lp_double_sig)
    bic_avg_params['gauss'] = estimate_bic(4,len_x,lp_gauss)
    bic_avg_params['sigmoidal'] = estimate_bic(4,len_x,lp_sig)
    bic_avg_params['uniform'] = estimate_bic(1,len_x,lp_unif)

    aic_avg_params['double sigmoidal'] = estimate_aic(7,lp_double_sig)
    aic_avg_params['gauss'] = estimate_aic(4,lp_gauss)
    aic_avg_params['sigmoidal'] = estimate_aic(4,lp_sig)
    aic_avg_params['uniform'] = estimate_aic(1,lp_unif)

    for sub in range(n_subset):
        bic_sub = {}
        idx_sub = sorted(np.random.choice(range(len(xdata)),int(len_x*perc_subset),replace=False))
        xdat_sub = xdata[idx_sub]
        len_xdat_sub = len(xdat_sub)
        lp_double_sig = log_probability_double_sigmoidal(theta_dict['double sigmoidal'], xdat_sub, 
                                         np.array([counts[x] for x in idx_sub]), 
                                         np.array([libsizes[x] for x in idx_sub]), median_counts, phi_opt, len_x, max_expr)
        lp_sig = log_probability_sigmoidal(theta_dict['sigmoidal'], xdat_sub, 
                                         np.array([counts[x] for x in idx_sub]), 
                                         np.array([libsizes[x] for x in idx_sub]), median_counts, phi_opt, len_x, max_expr)
        lp_gauss = log_probability_gaussian(theta_dict['gauss'], xdat_sub, 
                                         np.array([counts[x] for x in idx_sub]), 
                                         np.array([libsizes[x] for x in idx_sub]), median_counts, phi_opt, len_x, max_expr)
        lp_unif = log_probability_uniform(theta_dict['uniform'], xdat_sub, 
                                         np.array([counts[x] for x in idx_sub]), 
                                         np.array([libsizes[x] for x in idx_sub]), median_counts, phi_opt, max_expr)
        bic_subs_dict['double sigmoidal'].append(estimate_bic(7,len_xdat_sub,lp_double_sig))
        bic_subs_dict['gauss'].append(estimate_bic(4,len_xdat_sub,lp_gauss))
        bic_subs_dict['sigmoidal'].append(estimate_bic(4,len_xdat_sub,lp_sig))
        bic_subs_dict['uniform'].append(estimate_bic(1,len_xdat_sub,lp_unif))

    return bic_subs_dict,bic_avg_params,aic_avg_params

def measure_best_fits(samplers,info,ordered_expression,bic_subs_dict):
    mode_liks_full = info['mode_liks_full']
    max_args_full = info['max_args_full']
    poor_fits = []
    k_dict = {'double sigmoidal':7,'gauss':4,'sigmoidal':4,'uniform':1} 
    n=len(ordered_expression)
    p_overlap_dict = {}
    bic = {}
    inflec_points_idx = {'double sigmoidal':[3,4],'sigmoidal':[1],'gauss':[1,2],'uniform':[0]}
    for type_ in ['double sigmoidal','gauss','sigmoidal','uniform']:
        bic[type_] = estimate_bic(k_dict[type_],n,mode_liks_full[type_])
        sig_params = samplers[type_].get_chain(discard=5000,flat=False)[:,max_args_full[type_]]
        #only check inflection points are overlapping
        n_params = len(sig_params[0,0])
        p_overlap = []
        for kk,n_param in enumerate(range(n_params)):
            if kk not in inflec_points_idx[type_]: continue
            vals = []
            for j,sig_param in enumerate(sig_params[0,:]):
                vals.append(list(sig_params[:,j][:,n_param]))
            p_overlap.append(histogram_intersection(vals))
        if min(p_overlap) < 0.05:
            poor_fits.append(type_)
        p_overlap_dict[type_] = min(p_overlap)
    best_fit = 'uniform' #initiate best fit as uniform -- update if threshholds met
    inflection_points,inflection_points_2,inflection_point_derivs,inflection_point_derivs_2 = None,None,None,None
    if (max(bic_subs_dict['double sigmoidal']) - min(bic_subs_dict['uniform']) < 0) & \
        (max(bic_subs_dict['double sigmoidal']) - np.percentile(bic_subs_dict['gauss'],5) < 0) & \
        (max(bic_subs_dict['double sigmoidal']) - np.percentile(bic_subs_dict['sigmoidal'],5) < 0):
        best_fit = 'double sigmoidal'
        ## get flattened chains and estimate inflection points and derivs ##
        # double_sig params: b_min, b_max, b_final, x1, x2, k1, k2
        sig_params = samplers[best_fit].get_chain(discard=5000,flat=False)[:,max_args_full[best_fit]]
        sig_params = np.concatenate(sig_params,axis=0)
        inflection_points = sig_params[:,3]
        inflection_points_2 = sig_params[:,4]
        inflection_point_derivs = [double_sig_deriv(inflection_points[j], sig_params[:,0][j], sig_params[:,1][j],
                                                          sig_params[:,2][j],sig_params[:,3][j],sig_params[:,4][j],sig_params[:,5][j],
                                                          sig_params[:,6][j]) for j in range(len(inflection_points))]
        inflection_point_derivs_2 = [double_sig_deriv(inflection_points_2[j], sig_params[:,0][j], sig_params[:,1][j],
                                                          sig_params[:,2][j],sig_params[:,3][j],sig_params[:,4][j],sig_params[:,5][j],
                                                          sig_params[:,6][j]) for j in range(len(inflection_points_2))]
    elif (max(bic_subs_dict['sigmoidal']) - min(bic_subs_dict['uniform']) < 0) & \
        (np.mean(bic_subs_dict['sigmoidal']) - np.mean(bic_subs_dict['gauss']) < 0):
        best_fit = 'sigmoidal'
        ## get flattened chains and estimate inflection points and derivs ##
        # sig params: L, x0, k, b #
        sig_params = samplers[best_fit].get_chain(discard=5000,flat=False)[:,max_args_full[best_fit]]
        sig_params = np.concatenate(sig_params,axis=0)
        x0_flattened = sig_params[:,1]
        k_flattened = sig_params[:,2]
        inflection_points = x0_flattened
        inflection_point_derivs = [sig_deriv(inflection_points[j], sig_params[:,0][j], sig_params[:,1][j],
                                               sig_params[:,2][j],sig_params[:,3][j]) for j in range(len(inflection_points))]
    elif (max(bic_subs_dict['gauss']) - min(bic_subs_dict['uniform']) < 0) & \
        (np.mean(bic_subs_dict['gauss']) - np.mean(bic_subs_dict['sigmoidal']) < 0):
        best_fit = 'gauss'
        ## get flattened chains and estimate inflection points and derivs ##
        # gauss params: a, x0, sigma, y_offset #
        gauss_params = samplers[best_fit].get_chain(discard=5000,flat=False)[:,max_args_full[best_fit]]
        gauss_params = np.concatenate(gauss_params,axis=0)
        x0_flattened = gauss_params[:,1]
        sigma_flattened = gauss_params[:,2]
        inflection_points = x0_flattened-sigma_flattened
        inflection_points_2 = x0_flattened+sigma_flattened
        inflection_point_derivs = [gauss_deriv(inflection_points[j], gauss_params[:,0][j], gauss_params[:,1][j],
                                               gauss_params[:,2][j],gauss_params[:,3][j]) for j in range(len(inflection_points))]
        inflection_point_derivs_2 = [gauss_deriv(inflection_points_2[j], gauss_params[:,0][j], gauss_params[:,1][j],
                                               gauss_params[:,2][j],gauss_params[:,3][j]) for j in range(len(inflection_points_2))]
    info['best_fit'] = best_fit
    info['inflection_points'] = inflection_points
    info['inflection_points_2'] = inflection_points_2
    info['inflection_point_derivs'] = inflection_point_derivs
    info['inflection_point_derivs_2'] = inflection_point_derivs_2
    info['bic_mode_liks'] = bic
    info['poor_fits'] = poor_fits
    return  info

def annotate_useful_info(samplers,info):
    mode_liks_full = {}
    med_liks_full = {}
    acceptance_fraction = {}
    max_args = {}
    for type_ in ['double sigmoidal','gauss','sigmoidal','uniform']:
        acceptance_fraction[type_] = samplers[type_].acceptance_fraction
        n_sub = int(len(acceptance_fraction[type_])/2)
        max_args[type_] = np.argsort(acceptance_fraction[type_])[n_sub:]
        liks = np.ndarray.flatten(samplers[type_].get_log_prob(discard=5000)[:,max_args[type_]])
        try:
            mode_liks_full[type_] = get_mode(liks)
        except:
            mode_liks_full[type_] = np.nan
        try:
            med_liks_full[type_] = np.median(liks)
        except:
            med_liks_full[type_] = np.nan
        info['mode_liks_full'] = mode_liks_full
        info['max_args_full'] = max_args
        info['med_liks_full'] = med_liks_full
    return info

def run_autocorrelation_analysis(info,samplers):
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
    return autocorrelation_analysis

def comp_helper_simult(gene_relat,gene1,gene2,g1_deriv_mode,g2_deriv_mode):
    if g1_deriv_mode*g2_deriv_mode > 0: #same sign
        gene_relat[(gene2,gene1)] = '+'
        gene_relat[(gene1,gene2)] = '+'
    else: #different sign
        gene_relat[(gene1,gene2)] = '-'
        gene_relat[(gene2,gene1)] = '-'
        '''
        if g1_deriv_mode > 0:
            #gene_relat[(gene2,gene1)] = '+' ???
            gene_relat[(gene1,gene2)] = '-'
        else:
            #gene_relat[(gene2,gene1)] = '-' ???
            gene_relat[(gene1,gene2)] = '+'
        '''
    return gene_relat

def comp_helper_non_simult(gene_relat,gene1,gene2,g1_deriv_mode,g2_deriv_mode):
    if g1_deriv_mode < 0:
        if g2_deriv_mode < 0:
            gene_relat[(gene1,gene2)] = '+'
        else: #gene1 down then gene1 up
            gene_relat[(gene1,gene2)] = '-'
    else: # gene1 up
        if g2_deriv_mode < 0: #gene1 up then gene2 down
            gene_relat[(gene1,gene2)] = '-'
            gene_relat[(gene2,gene1)] = '+'
        else: #gene1 up then gene2 up
            gene_relat[(gene1,gene2)] = '+'
    return gene_relat

def compare_genes(gene1,gene2,inflection_points,inflection_point_derivs,
    inflection_points_2,inflection_point_derivs_2,gene1_mode1=True):
    #TODO: Consolidate code here
    gene_relat = {}
    if gene1_mode1:
        g1_inflec_points = inflection_points[gene1]
        g1_derivs = inflection_point_derivs[gene1]
    else:
        g1_inflec_points = inflection_points_2[gene1]
        g1_derivs = inflection_point_derivs_2[gene1]

    g1_mode = get_mode(g1_inflec_points)
    g1_deriv_mode = np.mean(g1_derivs)

    #first, compare gene1 inflection point to gene2 inflection point 1
    g2_mode1 = get_mode(inflection_points[gene2])
    g2_deriv_mode1 = np.mean(inflection_point_derivs[gene2])

    p_overlap1 = histogram_intersection([g1_inflec_points,inflection_points[gene2]])
    if p_overlap1 > 0.05: #strong overlap of g1 and g2 inflection point 1
        gene_relat = comp_helper_simult(gene_relat,gene1,gene2,g1_deriv_mode,g2_deriv_mode1)

    elif g1_mode < g2_mode1: #NOTE: we still don't know if g1_mode2 < g2_mode1
        gene_relat = comp_helper_non_simult(gene_relat,gene1,gene2,g1_deriv_mode,g2_deriv_mode1)

    else:
        if inflection_points_2[gene2] is not None:
            g2_mode2 = get_mode(inflection_points_2[gene2])
            g2_deriv_mode2 = np.mean(inflection_point_derivs_2[gene2])
            p_overlap2 = histogram_intersection([g1_inflec_points,inflection_points_2[gene2]])
            if p_overlap2 > 0.05: #strong overlap of g1 and g2 inflection point 1
                gene_relat = comp_helper_simult(gene_relat,gene1,gene2,g1_deriv_mode,g2_deriv_mode2)

            if g1_mode < g2_mode2:
                gene_relat = comp_helper_non_simult(gene_relat,gene1,gene2,g1_deriv_mode,g2_deriv_mode2)

            elif g2_mode2 < g1_mode:
                gene_relat = comp_helper_non_simult(gene_relat,gene2,gene1,g2_deriv_mode2,g1_deriv_mode)

    return gene_relat

def update_shared_relationships(gene_relationships,gene_relat_all):
    gene_relat_update = {}
    for gene_relat in gene_relat_all:
        for gene_check in gene_relat:
            if gene_check not in gene_relat_update:
                gene_relat_update[gene_check] = set([])
            gene_relat_update[gene_check].add(gene_relat[gene_check])
    for gene_check in gene_relat_update:
        if len(gene_relat_update[gene_check]) == 1:
            gene_relationships[gene_check] = list(gene_relat_update[gene_check])[0]
    return gene_relationships

def build_gene_relationships(genes_good_fit,best_fits,inflection_points,inflection_point_derivs,
    inflection_points_2,inflection_point_derivs_2,tfs=None):
    gene_relationships = {}
    for gene1,gene2 in itertools.combinations(genes_good_fit,2):
        if tfs is not None:
            if (gene1.upper() not in tfs) or (gene2.upper() not in tfs): continue
        if (best_fits[gene1] == 'uniform') or (best_fits[gene2] == 'uniform'): 
            continue
        gene_relat_all = []
        gene_relat_all.append(compare_genes(gene1,gene2,inflection_points,inflection_point_derivs,
            inflection_points_2,inflection_point_derivs_2,gene1_mode1=True))
        if inflection_points_2[gene1] is not None:
            gene_relat_all.append(compare_genes(gene1,gene2,inflection_points,inflection_point_derivs,
                inflection_points_2,inflection_point_derivs_2,gene1_mode1=False))
        gene_relat_all.append(compare_genes(gene2,gene1,inflection_points,inflection_point_derivs,
            inflection_points_2,inflection_point_derivs_2,gene1_mode1=True))
        if inflection_points_2[gene2] is not None:
            gene_relat_all.append(compare_genes(gene2,gene1,inflection_points,inflection_point_derivs,
                inflection_points_2,inflection_point_derivs_2,gene1_mode1=False))
        gene_relationships = update_shared_relationships(gene_relationships,gene_relat_all)
    return gene_relationships
