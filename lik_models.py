import pandas as pd
import scanpy as sc
import numpy as np
import pickle as pkl

import emcee
import math
from scipy.special import gamma
from scipy.special import loggamma

import sys
import os

def estimate_nb_log_likelihood(x,mu,phi):
    log2 = np.log(2)
    #NOTE: Gamma(n+1) = n!
    return loggamma(x+phi)/log2-loggamma(phi)/log2-loggamma(x+1)/log2+x*(np.log(mu)/log2-np.log(mu+phi)/log2)+\
phi*(np.log(phi)/log2-np.log(mu+phi)/log2)

def gaussian_pdf_likelihood(x,mu,sigma):
    log2 = np.log(2)
    return(-np.log(sigma*np.sqrt(2*np.math.pi))/log2 - 0.5*((x-mu)/sigma)**2*np.log(np.exp(1))/log2)

def likelihood_prior_L(x,lambda_,max_L):
    scaling_factor = lambda_*max_L+np.exp(-lambda_*max_L)-1
    log2 = np.log(2)
    return(np.log(lambda_*(1-np.exp(-lambda_*x))/scaling_factor)/log2)

def folded_normal_pdf_likelihood(x,mu,sigma):
    return np.log(1/np.sqrt(2*np.math.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))+\
        1/np.sqrt(2*np.math.pi*sigma**2)*np.exp(-(x+mu)**2/(2*sigma**2)))/np.log(2)

def exponential_pdf_likelihood(x,lambda_):
    log2 = np.log(2)
    return(-np.log(lambda_)/log2-lambda_*x*np.log(np.exp(1))/log2)

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

def double_sigmoid(x, b_min, b_mid, b_max, x1, x2, k1, k2):
    # see https://link-springer-com.ezproxy.molgen.mpg.de/article/10.1007/s10203-020-00279-7 for description
    y = b_min+(b_mid-b_min)/(1+np.exp(-k1*(x-x1)))+(b_max-b_mid)/(1+np.exp(-k2*(x-x2)))
    return (y)

def gaussian(x, a, x0, sigma, y_offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+y_offset

def double_sig_double_deriv(x,b_min, b_mid, b_max, x1, x2, k1, k2):
    return k1**2*(b_mid-b_min)*np.exp(-k1*(x-x1))*(np.exp(-k1*(x-x1))-1)/((1+np.exp(-k1*(x-x1)))**3)+\
    k2**2*(b_max-b_mid)*np.exp(-k2*(x-x2))*(np.exp(-k2*(x-x2))-1)/((1+np.exp(-k2*(x-x2)))**3)

def double_sig_double_deriv_inflec1(x,b_min, b_mid, b_max, x1, x2, k1, k2):
    return k1**2*(b_mid-b_min)*np.exp(-k1*(x-x1))*(np.exp(-k1*(x-x1))-1)/((1+np.exp(-k1*(x-x1)))**3)

def double_sig_double_deriv_inflec2(x,b_min, b_mid, b_max, x1, x2, k1, k2):
    return k2**2*(b_max-b_mid)*np.exp(-k2*(x-x2))*(np.exp(-k2*(x-x2))-1)/((1+np.exp(-k2*(x-x2)))**3)

def check_is_inflec_double_sigmoid_inflec1(params,dx=1):
    inflec = params[3]
    if np.isnan(double_sig_double_deriv(inflec-dx,*params)) or np.isnan(double_sig_double_deriv(inflec+dx,*params)):
        ## Note - this happens due to dividing inf by inf
        if np.isnan(double_sig_double_deriv_inflec1(inflec-dx,*params)) or np.isnan(double_sig_double_deriv_inflec1(inflec+dx,*params)):
            return(np.nan)
        else:
            if np.sign(double_sig_double_deriv_inflec1(inflec-dx,*params))*np.sign(double_sig_double_deriv_inflec1(inflec+dx,*params)) <= 0:
                return(True)
            else:
                return(False)
    if np.sign(double_sig_double_deriv(inflec-dx,*params))*np.sign(double_sig_double_deriv(inflec+dx,*params)) <= 0:
        return(True)
    else:
        return(False)

def check_is_inflec_double_sigmoid_inflec2(params,dx=1):
    inflec = params[4]
    if np.isnan(double_sig_double_deriv(inflec-dx,*params)) or np.isnan(double_sig_double_deriv(inflec+dx,*params)):
        ## Note - this happens due to dividing inf by inf
        if np.isnan(double_sig_double_deriv_inflec1(inflec-dx,*params)) or np.isnan(double_sig_double_deriv_inflec1(inflec+dx,*params)):
            return(np.nan)
        else:
            if np.sign(double_sig_double_deriv_inflec1(inflec-dx,*params))*np.sign(double_sig_double_deriv_inflec1(inflec+dx,*params)) <= 0:
                return(True)
            else:
                return(False)
    if np.sign(double_sig_double_deriv(inflec-dx,*params))*np.sign(double_sig_double_deriv(inflec+dx,*params)) <= 0:
        return(True)
    else:
        return(False)

def log_prior_uniform(theta, max_expr):
    if 0 < theta[0] <= max_expr:
        return np.log(1./max_expr)/np.log(2)
    return -np.inf

def log_prior_gaussian(theta, x, mu_arr_map_counts, len_full_data, max_expr):
    if min(mu_arr_map_counts) <= 0:
        return -np.inf
    a, x0, sigma, y_offset = theta
    if (0 < a <= max_expr) and (0 < y_offset <= max_expr) and (0 <= x0 <= len_full_data) and (sigma > 0):
        return folded_normal_pdf_likelihood(sigma,mu=0,sigma=0.1*len_full_data)+2*np.log(1./max_expr)/np.log(2)+np.log(1./len_full_data)/np.log(2)
    return -np.inf

def log_prior_sigmoid(theta, x, mu_arr_map_counts, len_full_data, max_expr):
    if min(mu_arr_map_counts) <= 0:
        return -np.inf
    L ,x0, k, b = theta
    max_counts = max(mu_arr_map_counts)
    if (0 < b <= max_expr) and (0 <= x0 <= len_full_data) and (0 < L <= max_expr):# and (L<=max_counts):
        return folded_normal_pdf_likelihood(k,mu=0,sigma=0.1)+2*np.log(1./max_expr)/np.log(2)+np.log(1./len_full_data)/np.log(2)
    return -np.inf

def log_prior_double_sigmoid(theta, x, mu_arr_map_counts, len_full_data, max_expr):
    if min(mu_arr_map_counts) <= 0:
        return -np.inf
    b_min, b_mid, b_max, x1, x2, k1, k2 = theta
    #ensure inflection point occurs x1 and x2 -- if not, retun -inf
    if not (check_is_inflec_double_sigmoid_inflec1(theta)):
        return -np.inf
    if not (check_is_inflec_double_sigmoid_inflec2(theta)):
        return -np.inf
    if (0 <= x1 < x2 <= len_full_data) and (0 < b_min <= max_expr) and (0 < b_mid <= max_expr) and (0 < b_max <= max_expr) and (k1 > 0) and (k2 > 0):
        return folded_normal_pdf_likelihood(k1,mu=0,sigma=0.1)+folded_normal_pdf_likelihood(k2,mu=0,sigma=0.1)+3*np.log(1./max_expr)/np.log(2)+2*np.log(1./len_full_data)/np.log(2)
    return -np.inf

def log_probability_uniform(theta, xdata, counts, libsizes, median_counts, phi_opt, max_expr):
    mu_arr = np.array([theta[0]]*len(xdata))
    mu_arr_map_counts = libsizes/median_counts*(np.exp(mu_arr)-1)
    lp = log_prior_uniform(theta, max_expr)
    if not np.isfinite(lp):
        return -np.inf
    lp_tot = lp + np.sum([estimate_nb_log_likelihood(counts,mu_arr_map_counts,phi_opt)])
    if np.isnan(lp_tot):
        print("This shouldn't happen.... debugging:")
        print("Uniform")
        print("mu:",mu_arr_map_counts)
        print("min mu:",min(mu_arr_map_counts))
        return -np.inf
    else:
        return lp_tot

def log_probability_gaussian(theta, xdata, counts, libsizes, median_counts, phi_opt, len_full_data, max_expr):
    mu_arr = gaussian(xdata, *theta)
    mu_arr_map_counts = libsizes/median_counts*(np.exp(mu_arr)-1)
    lp = log_prior_gaussian(theta, xdata, mu_arr_map_counts, len_full_data, max_expr)
    if not np.isfinite(lp):
        return -np.inf
    lp_tot = lp + np.sum([estimate_nb_log_likelihood(counts,mu_arr_map_counts,phi_opt)])
    if np.isnan(lp_tot):
        print("This shouldn't happen.... debugging:")
        print("Gaussian")
        print("mu:",mu_arr_map_counts)
        print("min mu:",min(mu_arr_map_counts))
        print("max mu:",max(mu_arr_map_counts))
        return -np.inf
    else:
        return lp_tot

def log_probability_sigmoidal(theta, xdata, counts, libsizes, median_counts, phi_opt, len_full_data, max_expr):
    mu_arr = sigmoid(xdata, *theta)
    mu_arr_map_counts = libsizes/median_counts*(np.exp(mu_arr)-1)
    lp = log_prior_sigmoid(theta, xdata, mu_arr_map_counts, len_full_data, max_expr)
    if not np.isfinite(lp):
        return -np.inf
    lp_tot = lp + np.sum([estimate_nb_log_likelihood(counts,mu_arr_map_counts,phi_opt)])
    if np.isnan(lp_tot):
        print("This shouldn't happen.... debugging:")
        print("Sigmoidal")
        print("mu:",mu_arr_map_counts)
        print("min mu:",min(mu_arr_map_counts))
        print("max mu:",max(mu_arr_map_counts))
        return -np.inf
    else:
        return lp_tot
    
def log_probability_double_sigmoidal(theta, xdata, counts, libsizes, median_counts, phi_opt, len_full_data, max_expr):
    mu_arr = double_sigmoid(xdata, *theta)
    mu_arr_map_counts = libsizes/median_counts*(np.exp(mu_arr)-1)
    lp = log_prior_double_sigmoid(theta, xdata, mu_arr_map_counts, len_full_data, max_expr)
    if not np.isfinite(lp):
        return -np.inf
    lp_tot = lp + np.sum([estimate_nb_log_likelihood(counts,mu_arr_map_counts,phi_opt)])
    if np.isnan(lp_tot):
        print("This shouldn't happen.... debugging:")
        print("Double Sigmoid")
        print("mu:",mu_arr_map_counts)
        print("min mu:",min(mu_arr_map_counts))
        print("max mu:",max(mu_arr_map_counts))
        return -np.inf
    else:
        return lp_tot