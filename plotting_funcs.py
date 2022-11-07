import numpy as np
import matplotlib.pyplot as plt
import corner
from lik_models import *
from helper_funcs import *
import os
import functools
import operator
from scipy.stats import gaussian_kde
import seaborn as sns
import pickle as pkl
from matplotlib import gridspec

def plot_final_curves(mcmc,gene,type_,n_discard=5000,use_max_args=True,figsize=(8,4),idx=-1):
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    sampler = samplers[type_]
    expr_dict = mcmc.expr_dict
    xdata = range(len(expr_dict[gene]))
    ordered_expression = np.array(expr_dict[gene])
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.plot(xdata, ordered_expression,'.', markerfacecolor='None',color='darkgray', lw=0.1, markersize=10, alpha=1,zorder=-1)
    plt.xlabel('Pseudotime Ordering',fontsize=18)
    plt.ylabel('Expression',fontsize=18)
    if type_ == 'gauss':
        for j,params in enumerate(samplers[type_].get_chain()[idx]):
            plot_ = True
            if use_max_args:
                if j not in mcmc.max_args_full[gene][type_]:
                    plot_ = False
            if plot_:
                plt.plot(xdata,gaussian(xdata,*params),label=str(j)+' '+str(samplers[type_].get_log_prob()[idx][j]))
    elif type_ == 'sigmoidal':
        for j,params in enumerate(samplers[type_].get_chain()[idx]):
            plot_ = True
            if use_max_args:
                if j not in mcmc.max_args_full[gene][type_]:
                    plot_ = False
            if plot_:
                plt.plot(xdata,sigmoid(xdata,*params),label=str(j)+' '+str(samplers[type_].get_log_prob()[idx][j]))
    elif type_ == 'double sigmoidal':
        for j,params in enumerate(samplers[type_].get_chain()[idx]):
            plot_ = True
            if use_max_args:
                if j not in mcmc.max_args_full[gene][type_]:
                    plot_ = False
            if plot_:
                plt.plot(xdata,double_sigmoid(xdata,*params),label=str(j)+' '+str(samplers[type_].get_log_prob()[idx][j]))
    elif type_ == 'uniform':
        for j,params in enumerate(samplers[type_].get_chain()[idx]):
            plot_ = True
            if use_max_args:
                if j not in mcmc.max_args_full[gene][type_]:
                    plot_ = False
            if plot_:
                plt.plot(xdata,[params[0]]*len(xdata),label=str(j)+' '+str(samplers[type_].get_log_prob()[idx][j]))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12,frameon=False)
    plt.title(gene+' '+type_+' fits',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return fig

def make_fit_plot(mcmc,gene,type_,n_discard=5000,use_max_args=True):
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    sampler = samplers[type_]
    expr_dict = mcmc.expr_dict
    xdata = range(len(expr_dict[gene]))
    ordered_expression = np.array(expr_dict[gene])
    fig = plt.figure(figsize=(8,4))
    if use_max_args:
        flat_samples = sampler.get_chain(discard=n_discard, flat=False)[:,mcmc.max_args_full[gene][type_]]
    else:
        flat_samples = sampler.get_chain(discard=n_discard, flat=False)
    flat_samples = np.concatenate(flat_samples,axis=0)
    inds = np.random.randint(len(flat_samples), size=100)
    plt.plot(xdata, ordered_expression,'.', markerfacecolor='None',color='darkgray', lw=0.1, markersize=10, alpha=1,zorder=-1)
    plt.xlabel('Pseudotime Ordering',fontsize=18)
    plt.ylabel('Expression',fontsize=18)
    for ind in inds:
        sample = flat_samples[ind]
        if type_ == 'gauss':
            func_ = gaussian(xdata, *sample)
        elif type_ == 'sigmoidal':
            func_ = sigmoid(xdata, *sample)
        elif type_ == 'double sigmoidal':
            func_ = double_sigmoid(xdata, *sample)
        else:
            func_ = np.array([sample[0]]*len(xdata))
        plt.plot(xdata,func_, '-',color='blue',linewidth=1,alpha=0.1)
    plt.title(gene+', '+type_ + ' fit',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return fig

def make_joint_fit_plot(mcmc,gene,n_discard=5000,use_max_args=True):
    label_dict = {'gauss':'Gaussian','sigmoidal':'sigmoidal',
                 'double sigmoidal':'double sigmoidal','uniform':'uniform'}
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    expr_dict = mcmc.expr_dict
    clrs = sns.color_palette('Set1', n_colors=10)  # a list of RGB tuples
    color_dict = {'gauss':clrs[0],'sigmoidal':clrs[1],'double sigmoidal':clrs[2],'uniform':clrs[3]}
    fig = plt.figure(figsize=(8,6))
    xdata = range(len(expr_dict[gene]))
    plt.plot(xdata, expr_dict[gene],'.', markerfacecolor='None',color='darkgray', lw=0.1, markersize=10, alpha=1,zorder=-1)
    plt.title(gene,fontsize=20)
    plt.xlabel('Pseudotime Ordering',fontsize=18)
    plt.ylabel('Expression',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for type_ in ['gauss','sigmoidal','double sigmoidal','uniform']:
        sampler = samplers[type_]
        if use_max_args:
            flat_samples = sampler.get_chain(discard=n_discard, flat=False)[:,mcmc.max_args_full[gene][type_]]
        else:
            flat_samples = sampler.get_chain(discard=n_discard, flat=False)
        sig_params = np.concatenate(flat_samples,axis=0)
        x0_flattened = flat_samples[:,1]
        inds = np.random.randint(len(x0_flattened), size=10)
        for ind in inds:
            sample = x0_flattened[ind]
            if type_ == 'gauss':
                func_ = gaussian(xdata, *sample)
            elif type_ == 'sigmoidal':
                func_ = sigmoid(xdata, *sample)
            elif type_ == 'double sigmoidal':
                func_ = double_sigmoid(xdata, *sample)
            else:
                func_ = np.array([sample[0] for x in xdata])
            plt.plot(xdata,func_, '-',color=color_dict[type_],linewidth=2,alpha=0.5)
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    for type_ in ['double sigmoidal','sigmoidal','gauss','uniform']:
        plt.plot([-1000],[0], '-',color=color_dict[type_],linewidth=5,alpha=1,label=label_dict[type_])
    plt.legend(fontsize=16)
    return fig

def make_joint_fit_and_bic_estimate_plot(mcmc,gene,n_discard=5000,use_max_args=True):
    label_dict = {'gauss':'Gaussian','sigmoidal':'sigmoidal',
                 'double sigmoidal':'double sigmoidal','uniform':'uniform'}
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    expr_dict = mcmc.expr_dict
    clrs = sns.color_palette('Set1', n_colors=10)  # a list of RGB tuples
    color_dict = {'gauss':clrs[0],'sigmoidal':clrs[1],'double sigmoidal':clrs[2],'uniform':clrs[3]}
    fig = plt.figure(figsize=(13,6))
    ax = plt.subplot(121)
    xdata = range(len(expr_dict[gene]))
    plt.plot(xdata, expr_dict[gene],'.', markerfacecolor='None',color='darkgray', lw=0.1, markersize=10, alpha=1,zorder=-1)
    plt.title(gene+' MCMC fits',fontsize=20)
    plt.xlabel('Pseudotime Ordering',fontsize=18)
    plt.ylabel('Expression',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    for type_ in ['gauss','sigmoidal','double sigmoidal','uniform']:
        sampler = samplers[type_]
        if use_max_args:
            flat_samples = sampler.get_chain(discard=n_discard, flat=False)[:,mcmc.max_args_full[gene][type_]]
        else:
            flat_samples = sampler.get_chain(discard=n_discard, flat=False)
        sig_params = np.concatenate(flat_samples,axis=0)
        x0_flattened = flat_samples[:,1]
        inds = np.random.randint(len(x0_flattened), size=10)
        for ind in inds:
            sample = x0_flattened[ind]
            if type_ == 'gauss':
                func_ = gaussian(xdata, *sample)
            elif type_ == 'sigmoidal':
                func_ = sigmoid(xdata, *sample)
            elif type_ == 'double sigmoidal':
                func_ = double_sigmoid(xdata, *sample)
            else:
                func_ = np.array([sample[0] for x in xdata])
            plt.plot(xdata,func_, '-',color=color_dict[type_],linewidth=2,alpha=0.5)
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    for type_ in ['double sigmoidal','sigmoidal','gauss','uniform']:
        plt.plot([-1000],[0], '-',color=color_dict[type_],linewidth=5,alpha=1,label=label_dict[type_])
    plt.legend(fontsize=16)
    
    ax = plt.subplot(122)
    for type_ in ['double sigmoidal','sigmoidal','gauss','uniform']:#,'uniform']:
        sns.distplot(samplers['bic_subs_dict'][type_],color=color_dict[type_],label=label_dict[type_])

    plt.title(gene+' BIC estimates on 98% subsets',fontsize=20)
    plt.xlabel('BIC',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.0, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18, handletextpad=0.5,edgecolor='None', framealpha=0)
    plt.tight_layout()
    return fig

def make_mcmc_trace_plot(mcmc,gene,type_,sub_not_include=False,sub_in_red=False,n_discard=5000,plot_burnin=True):
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    sampler = samplers[type_]
    ndim = sampler.ndim
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    if type_ == 'gauss': labels = ["$a$", "$x_0$", "$\sigma$", "$b$"]
    elif type_ == 'sigmoidal': labels = ["$L$" ,"$x0$", "$k$", "$b$"]
    elif type_ == 'double sigmoidal': labels = ["$b_{min}$", "$b_{mid}$", "$b_{max}$", "$x_1$", "$x_2$", "$k_1$", "$k_2$"]
    elif type_ == 'uniform': labels = ["$b$"]
    else: return None
    title_dict = {'gauss':'Gaussian','sigmoidal':'Sigmoidal','double sigmoidal':'Double Sigmoidal','uniform':'Uniform'}
    for i in range(ndim):
        try:
            ax = axes[i]
            for j in range(len(sampler.get_chain()[0])):
                color = 'black'
                if j not in mcmc.max_args_full[gene][type_]:
                    if sub_in_red:
                        color='red'
                    if sub_not_include:
                        continue
                ax.plot(samples[:, j, i], color, alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.yaxis.set_label_coords(-0.1, 0.5)
        except:
            plt.plot(samples[:, :, i], 'k', alpha=0.3)
        ax.set_ylabel(labels[i],fontsize=18)
        ax.tick_params(axis='y', labelsize=14)
        if i == 0:
            ax.set_title(gene+' '+title_dict[type_]+' MCMC Trace Plot',fontsize=18)
    if plot_burnin:
        for i in range(ndim):
            ax = axes[i]
            if i == ndim-1:
                ax.axvline(x=n_discard,ymin=-0.2,ymax=1,c="black",linewidth=1,linestyle='--',zorder=0, clip_on=False)
            else:
                ax.axvline(x=n_discard,ymin=-1,ymax=1,c="black",linewidth=1,linestyle='--',zorder=0, clip_on=False)
    plt.xlabel("step number",fontsize=18)
    plt.xticks(fontsize=16)
    return fig

def make_corner_plot(mcmc,gene,type_,n_discard=5000,use_max_args=True):
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    sampler = samplers[type_]
    if use_max_args:
        flat_samples = sampler.get_chain(discard=n_discard, flat=False)[:,mcmc.max_args_full[gene][type_]]
    else:
        flat_samples = sampler.get_chain(discard=n_discard, flat=False)
    flat_samples = np.concatenate(flat_samples,axis=0)
    if type_ == 'gauss': labels = ["$a$", "$x_0$", "$\sigma$", "$b$"]
    elif type_ == 'sigmoidal': labels = ["$L$" ,"$x0$", "$k$", "$b$"]
    elif type_ == 'double sigmoidal': labels = ["$b_{min}$", "$b_{mid}$", "$b_{max}$", "$x_1$", "$x_2$", "$k_1$", "$k_2$"]
    elif type_ == 'uniform': labels = ["$b$"]
    else: return None
    meds = np.median(flat_samples,axis=0)
    modes = [get_mode(flat_samples[:,j]) for j in range(len(flat_samples[0]))]
    means = [np.mean(flat_samples[:,j]) for j in range(len(flat_samples[0]))]
    fig = corner.corner(flat_samples, labels=labels, truths=means, label_kwargs={'fontsize':18})
    title_dict = {'gauss':'Gaussian parameter estimates','sigmoidal':'sigmoidal parameter estimates',
                 'double sigmoidal':'double sigmoidal parameter estimates','uniform':'uniform parameter estimates'}
    if type_ == 'uniform':
        plt.title(gene+' '+title_dict[type_],fontsize=18)
    else:
        plt.subplots_adjust(top=0.92)
        plt.suptitle(gene+' '+title_dict[type_],fontsize=20)
    return fig

def add_violin(fig,data,iter_,color='gray',violin_cutoff=True, height=0.5):
    x_in = np.linspace(min(data), max(data), 101)
    density = gaussian_kde(data)
    density._compute_covariance()
    y = density(x_in)
    y_new = y / np.sum(y)
    left_sum = 0
    left_idx = 0
    for i in range(len(x_in)):
        left_sum += y_new[i]
        if left_sum >= 0.01: break
        else: left_idx = i
    right_sum = 0
    right_idx = len(x_in)-1
    for i in range(len(x_in)-1,-1,-1):
        right_sum += y_new[i]
        if right_sum >= 0.01: break
        else: right_idx = i
    y_new = y_new / max(y_new) * height
    if not violin_cutoff:
        left_idx,right_idx = 0,len(y)
    plt.fill_between(x_in[left_idx:right_idx],y_new[left_idx:right_idx] + iter_,
                     [iter_]*len(y_new[left_idx:right_idx]), color=color,
                     alpha=0.2, lw=2)
    return fig

def make_lik_hist_plot(mcmc, gene, n_discard=5000,use_max_args=True):
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    clrs = sns.color_palette('Set1', n_colors=10)  # a list of RGB tuples
    color_dict = {'gauss':clrs[0],'sigmoidal':clrs[1],'double sigmoidal':clrs[2],'uniform':clrs[3]}
    fig = plt.figure(figsize=(8,6))
    label_dict = {'gauss':'Gaussian','sigmoidal':'sigmoidal',
                 'double sigmoidal':'double sigmoidal','uniform':'uniform'}
    for type_ in ['double sigmoidal','sigmoidal','gauss','uniform']:
        if use_max_args:
            sns.distplot(np.ndarray.flatten(samplers[type_].get_log_prob(discard=n_discard)[:,mcmc.max_args_full[gene][type_]]),
                         label=label_dict[type_],color=color_dict[type_],)
        else:
            sns.distplot(np.ndarray.flatten(samplers[type_].get_log_prob(discard=n_discard)),
                         label=label_dict[type_],color=color_dict[type_])
    plt.ylim(0,2)
    plt.xlabel('log-likelihood',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(gene+ ' MCMC fits log-likelihood estimates',fontsize=20)
    plt.legend(fontsize=16)
    return fig

def make_acceptance_fraction_plot(mcmc,gene):
    samplers = pkl.load(open(mcmc.pkl_dir+gene+'.pkl','rb'))
    clrs = sns.color_palette('Set1', n_colors=10)  # a list of RGB tuples
    color_dict = {'gauss':clrs[0],'sigmoidal':clrs[1],'double sigmoidal':clrs[2],'uniform':clrs[3]}
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    full_vals = []
    pos = [0,1,2,3]
    colors_points = []
    colors = []
    keep_arg_ = []
    types_arg_ = []
    val_arg_ = []
    for type_ in ['double sigmoidal','sigmoidal','gauss','uniform']:
        color = []
        full_vals.append(samplers[type_].acceptance_fraction)
        val_arg_.extend(samplers[type_].acceptance_fraction)
        for j in range(len(samplers[type_].acceptance_fraction)):
            if j in mcmc.max_args_full[gene][type_]:
                color.append('black')
                keep_arg_.append('kept')
            else:
                color.append('red')
                keep_arg_.append('pruned')
            types_arg_.append(type_)
        colors_points.append(color)
        colors.append(color_dict[type_])
    data = {'type_':types_arg_,'keep':keep_arg_,'acceptance_rate':val_arg_}
    plot_df = pd.DataFrame.from_dict(data)
    ax = sns.swarmplot(x="type_", y="acceptance_rate", hue="keep", data=plot_df, 
                      order=['double sigmoidal','sigmoidal','gauss','uniform'],
                      edgecolor='black', hue_order=['kept','pruned'])
    bp = ax.boxplot(full_vals, positions=pos, showfliers=False)
    for element in ['medians']:
        plt.setp(bp[element], color='black')
    plt.xticks([0,1,2,3],['double\nsigmoidal','sigmoidal','Gaussian','uniform'],fontsize=18)
    plt.ylabel('Acceptance Rate [%]',fontsize=18)
    plt.yticks(fontsize=16)
    plt.title(gene+' Acceptance Rates',fontsize=20)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18,frameon=False)
    plt.xlabel('')
    return fig

def plot_autocorr_time_esimates(mcmc,gene,autocorrelations):
    
    auto_corr_length_int = autocorrelations[gene]['auto_corr_length_int']
    if mcmc.best_fits[gene] == 'double sigmoidal':
        labels = ["$b_{min}$", "$b_{mid}$", "$b_{max}$", "$x_1$", "$x_2$", "$k_1$", "$k_2$"]
        n_params = 7
    elif mcmc.best_fits[gene] == 'sigmoidal':
        labels = ["$L$" ,"$x0$", "$k$", "$b$"]
        n_params = 4
    elif mcmc.best_fits[gene] == 'gauss':
        labels = ["$a$", "$x_0$", "$\sigma$", "$b$"]
        n_params = 4
    else:
        labels = ["$b$"]
        n_params = 1
    
    fig = plt.figure(figsize=(n_params*2,2))
    ymin = 0
    ymax = 0
    for j,auto_corr in enumerate(auto_corr_length_int):
        for mcmc_iter in range(len(auto_corr[0])):
            if min([x[mcmc_iter] for x in auto_corr]) < ymin:
                ymin = min([x[mcmc_iter] for x in auto_corr])
            if max([x[mcmc_iter] for x in auto_corr]) > ymax:
                ymax = max([x[mcmc_iter] for x in auto_corr])

    autocorr_time_estimates = []
    for j,auto_corr in enumerate(auto_corr_length_int):
        plt.subplot(1,n_params,j+1)
        for mcmc_iter in range(len(auto_corr[0])):
            plt.plot(range(1,1000),[x[mcmc_iter] for x in auto_corr],color='lightgray',alpha=0.5)
        plt.plot(range(1,1000),np.mean(auto_corr,axis=1),color='black',alpha=1)
        plt.ylim(ymin-(ymax-ymin)*0.1,ymax+(ymax-ymin)*0.1)
        plt.title(labels[j])
        plt.xlabel(r"$T$")
        if j != 0: plt.yticks([])
        else: plt.ylabel(r"1+2$\sum_{\tau=1}^{T}\rho(\tau)$",fontsize=14)
        plt.axhline(max(np.mean(auto_corr,axis=1)),linestyle='--',color='k',linewidth=1)
        autocorr_time_estimates.append(max(np.mean(auto_corr,axis=1)))
    
    return fig
    
def plot_autocorrelation_func(mcmc,gene,autocorrelations):
    
    acf_params_tot = autocorrelations[gene]['acf_params_tot']
    if mcmc.best_fits[gene] == 'double sigmoidal':
        labels = ["$b_{min}$", "$b_{mid}$", "$b_{max}$", "$x_1$", "$x_2$", "$k_1$", "$k_2$"]
        n_params = 7
    elif mcmc.best_fits[gene] == 'sigmoidal':
        labels = ["$L$" ,"$x0$", "$k$", "$b$"]
        n_params = 4
    elif mcmc.best_fits[gene] == 'gauss':
        labels = ["$a$", "$x_0$", "$\sigma$", "$b$"]
        n_params = 4
    else:
        labels = ["$b$"]
        n_params = 1

    fig = plt.figure(figsize=(n_params*2,2))
    for j,acfs in enumerate(acf_params_tot):
        plt.subplot(1,n_params,j+1)
        for i in range(len(acfs[0])):
            plt.plot([x[i] for x in acfs],color='lightgray',alpha=0.5)
        plt.plot(np.mean(acfs,axis=1),color='black',alpha=1)
        plt.title(labels[j])
        plt.xlabel(r"$\tau$")
        plt.ylim(-0.4,1.1)
        if j != 0: plt.yticks([])
        else: plt.ylabel(r"$\rho(\tau)$",fontsize=14)
        plt.axhline(0,color='black',linestyle='--',linewidth=1)
    
    return fig

def make_inflec_point_comp_plot(mcmc,gene1,gene2,n_discard=5000,ylim1=None,ylim2=None,yticks1=None,yticks2=None):
    clrs = sns.color_palette('Set1', n_colors=10)  # a list of RGB tuples
    color_dict = {'gauss':clrs[0],'sigmoidal':clrs[1],'double sigmoidal':clrs[2],'uniform':clrs[3]}
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(4,12,wspace=1.8, hspace=0.2)
    ax = plt.subplot(gs[:2,:4])
    samplers = pkl.load(open(mcmc.pkl_dir+gene1+'.pkl','rb'))
    type_ = mcmc.best_fits[gene1]
    sampler = samplers[type_]
    expr_dict = mcmc.expr_dict
    xdata = range(len(expr_dict[gene1]))
    ordered_expression = np.array(expr_dict[gene1])
    flat_samples = sampler.get_chain(discard=n_discard, flat=False)[:,mcmc.max_args_full[gene1][type_]]
    flat_samples = np.concatenate(flat_samples,axis=0)
    inds = np.random.randint(len(flat_samples), size=100)
    plt.plot(xdata, ordered_expression,'.', markerfacecolor='None',color='darkgray', lw=0.1, markersize=5, alpha=1,zorder=-1)
    for ind in inds:
        sample = flat_samples[ind]
        if type_ == 'gauss':
            func_ = gaussian(xdata, *sample)
        elif type_ == 'sigmoidal':
            func_ = sigmoid(xdata, *sample)
        elif type_ == 'double sigmoidal':
            func_ = double_sigmoid(xdata, *sample)
        else:
            func_ = np.array([sample[0]]*len(xdata))
        plt.plot(xdata,func_, '-',color=color_dict[type_],linewidth=1,alpha=0.1)
    #plt.title(gene+', '+type_ + ' fit',fontsize=20)
    plt.xticks([])
    plt.yticks(fontsize=14)
    if ylim1 is not None:
        plt.ylim(ylim1[0],ylim1[1])
    #ax.yaxis.tick_right()
    plt.yticks(fontsize=14)
    plt.ylabel(gene1,fontsize=14)
    plt.title('MCMC fits',fontsize=14)
    if np.mean(mcmc.inflection_point_derivs[gene1]) > 0:
        if type_ == 'double sigmoidal': color1_gene1 = 'salmon'
        else: color1_gene1='red'
    else:
        if type_ == 'double sigmoidal': color1_gene1 = 'dodgerblue'
        else: color1_gene1='blue'
    ax.axvspan(min(mcmc.inflection_points[gene1]),max(mcmc.inflection_points[gene1]),alpha=0.3, color=color1_gene1)
    if mcmc.inflection_points_2[gene1] is not None:
        if np.mean(mcmc.inflection_point_derivs_2[gene1]) > 0:
            if type_ == 'double sigmoidal': color2_gene1 = 'salmon'
            else: color2_gene1='red'
        else:
            if type_ == 'double sigmoidal': color2_gene1 = 'dodgerblue'
            else: color2_gene1='blue'
        ax.axvspan(min(mcmc.inflection_points_2[gene1]),max(mcmc.inflection_points_2[gene1]),alpha=0.3, color=color2_gene1)

    ax = plt.subplot(gs[2:,:4])
    samplers = pkl.load(open(mcmc.pkl_dir+gene2+'.pkl','rb'))
    type_ = mcmc.best_fits[gene2]
    sampler = samplers[type_]
    expr_dict = mcmc.expr_dict
    xdata = range(len(expr_dict[gene2]))
    ordered_expression = np.array(expr_dict[gene2])
    flat_samples = sampler.get_chain(discard=n_discard, flat=False)[:,mcmc.max_args_full[gene2][type_]]
    flat_samples = np.concatenate(flat_samples,axis=0)
    inds = np.random.randint(len(flat_samples), size=100)
    plt.plot(xdata, ordered_expression,'.', markerfacecolor='None',color='darkgray', lw=0.1, markersize=5, alpha=1,zorder=-1)
    plt.xlabel('Pseudotime Ordering',fontsize=14)
    for ind in inds:
        sample = flat_samples[ind]
        if type_ == 'gauss':
            func_ = gaussian(xdata, *sample)
        elif type_ == 'sigmoidal':
            func_ = sigmoid(xdata, *sample)
        elif type_ == 'double sigmoidal':
            func_ = double_sigmoid(xdata, *sample)
        else:
            func_ = np.array([sample[0]]*len(xdata))
        plt.plot(xdata,func_, '-',color=color_dict[type_],linewidth=1,alpha=0.1)
    #plt.title(gene+', '+type_ + ' fit',fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if ylim2 is not None:
        plt.ylim(ylim2[0],ylim2[1])
    #ax.yaxis.tick_right()
    plt.yticks(fontsize=14)
    plt.ylabel(gene2,fontsize=14)
    if np.mean(mcmc.inflection_point_derivs[gene2]) > 0:
        if type_ == 'double sigmoidal': color1_gene2 = 'salmon'
        else: color1_gene2='red'
    else:
        if type_ == 'double sigmoidal': color1_gene2 = 'dodgerblue'
        else: color1_gene2='blue'
    ax.axvspan(min(mcmc.inflection_points[gene2]),max(mcmc.inflection_points[gene2]),alpha=0.3, color=color1_gene2)
    if mcmc.inflection_points_2[gene2] is not None:
        if np.mean(mcmc.inflection_point_derivs_2[gene2]) > 0:
            if type_ == 'double sigmoidal': color2_gene2 = 'salmon'
            else: color2_gene2='red'
        else:
            if type_ == 'double sigmoidal': color2_gene2 = 'dodgerblue'
            else: color2_gene2='blue'
        ax.axvspan(min(mcmc.inflection_points_2[gene2]),max(mcmc.inflection_points_2[gene2]),alpha=0.3, color=color2_gene2)

    ax = plt.subplot(gs[:,4:8])
    sns.distplot(mcmc.inflection_points[gene1],color=color1_gene1,label=gene1)#bins=n_bins
    sns.distplot(mcmc.inflection_points[gene2],color=color1_gene2,label=gene2)#bins=n_bins
    plt.xlabel('Pseudotime',fontsize=14)
    plt.xticks(fontsize=14)
    print('Inflec 1 overlap: ',histogram_intersection([mcmc.inflection_points[gene2],mcmc.inflection_points[gene1]],n_bins=100))
    plt.title('Inflection Point 1, p='+str(round(histogram_intersection([mcmc.inflection_points[gene2],
                                                                         mcmc.inflection_points[gene1]],n_bins=100),3)),fontsize=14)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.ylabel('Density',fontsize=14)
    ax.yaxis.tick_right()
    if yticks1 is not None:
        plt.yticks(yticks1)
    else:
        plt.yticks([x/100 for x in range(0,int(plt.ylim()[1]*100)+1)])
    plt.yticks(fontsize=14)
    plt.legend(frameon=False,fontsize=12)

    ax = plt.subplot(gs[:,8:])
    if mcmc.inflection_points_2[gene1] is not None:
        sns.distplot(mcmc.inflection_points_2[gene1],color=color2_gene1,label=gene1)#bins=n_bins
    if mcmc.inflection_points_2[gene2] is not None:
        sns.distplot(mcmc.inflection_points_2[gene2],color=color2_gene2,label=gene2)#bins=n_bins
    plt.xlabel('Pseudotime',fontsize=14)
    plt.ylabel('')
    plt.xticks(fontsize=14)
    if (mcmc.inflection_points_2[gene1] is not None) and (mcmc.inflection_points_2[gene2] is not None):
        print('Inflec 2 overlap: ',histogram_intersection([mcmc.inflection_points_2[gene2],mcmc.inflection_points_2[gene1]],n_bins=100))
        plt.title('Inflection Point 2, p='+str(round(histogram_intersection([mcmc.inflection_points_2[gene2],
                                                                             mcmc.inflection_points_2[gene1]],n_bins=100),3)),fontsize=14)
    else:
        plt.title('Inflection Point 2',fontsize=14)
    ax.yaxis.tick_right()
    if yticks2 is not None:
        plt.yticks(yticks2)
    else:
        plt.yticks([x/100 for x in range(0,int(plt.ylim()[1]*100)+1)])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.yticks(fontsize=14)
    plt.legend(frameon=False,fontsize=12)
    return fig

def plot_transcriptional_cascade(mcmc,title=None,xlim=None, figsize=(7,10), tfs=None, violin_cutoff=True, height=0.5):
    fig = plt.figure(figsize=(figsize[0],figsize[1]))
    ax = plt.subplot(111)
    iter_ = 0
    gene_labs = []
    non_unif_fits = {}
    for gene in mcmc.inflection_points:
        if mcmc.best_fits[gene] != 'uniform':
            non_unif_fits[gene] = mcmc.inflection_points[gene]
    for gene in [y[0] for y in sorted(non_unif_fits.items(),key = lambda x:np.mean(x[1]))]:
        mcmc.ordered_expression = mcmc.expr_dict[gene]
        if tfs is not None:
            if gene not in tfs: continue
        mode = np.mean(mcmc.inflection_points[gene])
        if mcmc.best_fits[gene] == 'gauss':
            mode2 = np.mean(mcmc.inflection_points_2[gene])
            deriv_mode1 = np.mean(mcmc.inflection_point_derivs[gene])
            deriv_mode2 = np.mean(mcmc.inflection_point_derivs_2[gene])
            if mode > 0:
                if deriv_mode1 > 0:
                    color='red'
                else:
                    color='blue'
                plt.plot(mode,iter_,'o',color=color,markersize=3)
                fig = add_violin(fig,mcmc.inflection_points[gene],iter_,color,violin_cutoff,height)
                mode_new = mode
            else:
                mode_new = 0
            if mode2 < len(mcmc.ordered_expression):
                if deriv_mode2 > 0:
                    color='red'
                else:
                    color='blue'
                plt.plot(mode2,iter_,'o',color=color,markersize=3)
                fig = add_violin(fig,mcmc.inflection_points_2[gene],iter_,color,violin_cutoff,height)
                mode2_new = mode2
            else:
                mode2_new = len(mcmc.ordered_expression)
            plt.plot([mode_new,mode2_new],[iter_,iter_],'--',color='darkgray',zorder=1,linewidth=1)
        elif mcmc.best_fits[gene] == 'double sigmoidal':
            mode2 = np.mean(mcmc.inflection_points_2[gene])
            deriv_mode1 = np.mean(mcmc.inflection_point_derivs[gene])
            deriv_mode2 = np.mean(mcmc.inflection_point_derivs_2[gene])
            if deriv_mode1 > 0:
                color = 'salmon'
            else:
                color = 'dodgerblue'
            plt.plot(mode,iter_,'o',color=color,markersize=3)
            fig = add_violin(fig,mcmc.inflection_points[gene],iter_,color,violin_cutoff,height)
            if deriv_mode2 > 0:
                color = 'salmon'
            else:
                color='dodgerblue'
            plt.plot(mode2,iter_,'o',color=color,markersize=3)
            fig = add_violin(fig,mcmc.inflection_points_2[gene],iter_,color,violin_cutoff,height)
            plt.plot([mode,mode2],[iter_,iter_],'--',color='darkgray',zorder=1,linewidth=1)
        else:
            deriv_mode = get_mode(mcmc.inflection_point_derivs[gene])
            if deriv_mode > 0:
                plt.plot(mode,iter_,'o',color='red',markersize=3)
                fig = add_violin(fig,mcmc.inflection_points[gene],iter_,'red',violin_cutoff,height)
            else:
                plt.plot(mode,iter_,'o',color='blue',markersize=3)
                fig = add_violin(fig,mcmc.inflection_points[gene],iter_,'blue',violin_cutoff,height)
        iter_ -= 1
        gene_labs.append(gene)
    plt.yticks(range(0,iter_,-1),gene_labs,fontsize=14)
    for yy in range(0,iter_,-1):
        plt.axhline(yy,linewidth=0.2,color='silver',zorder=-10,alpha=0.3)
    for xx in plt.xticks()[0]:
        plt.axvline(xx,linewidth=0.2,color='silver',zorder=-10,alpha=0.3)
    ax.yaxis.tick_right()
    if title is not None:
        plt.title(title,fontsize=24)
    else:
        plt.title('Transcriptional Cascades '+mcmc.name,fontsize=20)
    plt.xlabel('Pseudotime',fontsize=18)
    plt.xticks(fontsize=16)
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    return fig