import matplotlib
matplotlib.use('Agg')

import glob
import numpy as np
import os
import pickle
import argparse
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import util_classes
import util_classes_cheap
import util_tests_cheap
import sys

# plt.style.use('seaborn-v0_8-white')
if sys.version_info < (3, 9):
    plt.style.use('seaborn-white')
else:
    plt.style.use('seaborn-v0_8-white')

def plot_line(rejection_rate, rejection_rate_upper, rejection_rate_lower, times, labels, legend_text, marker, markersize, color, linestyle, xytext, log_time_scale, small_times):
    if log_time_scale:
        plt.semilogx(times, rejection_rate, label=legend_text, marker=marker, markersize=markersize, color=color, linestyle=linestyle)
    else:
        plt.plot(times, rejection_rate, label=legend_text, marker=marker, markersize=markersize, color=color, linestyle=linestyle)
        #if small_times:
            #plt.xlim([0,1200])
    plt.fill_between(times, rejection_rate_lower, rejection_rate_upper, alpha=.3, color=color)
    for i in range(len(rejection_rate_lower)):
        print(f'rejection_rate_upper[i]-rejection_rate_lower[i]: {rejection_rate_upper[i]-rejection_rate_lower[i]}, rejection_rate[i]: {rejection_rate[i]}')
    
    for x,y,z in zip(times, rejection_rate, labels):
        plt.annotate(z, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=xytext, # distance from text to points (x,y)
                     ha='center',
                     size=6) #,fontweight='bold') # horizontal alignment can be left, right or center
        
def pplot(ax=None):
    if ax is None:
        plt.grid(True, alpha=0.5)
        axoff(plt.gca())
    else:
        ax.grid(True, alpha=0.5)
        axoff(ax)
    return

def axoff(ax, keys=['top', 'right']):
    for k in keys:
        ax.spines[k].set_visible(False)
    return

def fig_rejection_two_sample(args, test_groups, joint_group_results_dict):
    
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    #['>', 's', 'o', 'D', '+', '*',  '>', 's', 'o', 'D', '>', 's', 'o', 'D']*2
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','red', 'black']*2
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)
    
    #Compute rejection rates, rejection rate bars, times and labels for Complete
    if not no_compute['complete']:
        rejection_rate['complete'], rejection_rate_upper['complete'], rejection_rate_lower['complete'], times['complete'], labels['complete'] = joint_group_results_dict['complete'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
        
    #Compute rejection rates, rejection rate bars, times and labels for Cheap Permutations
    if not no_compute['cheap_perm']:
        rejection_rate['cheap_perm'], rejection_rate_upper['cheap_perm'], rejection_rate_lower['cheap_perm'], times['cheap_perm'], labels['cheap_perm'] = joint_group_results_dict['cheap_perm'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
        
    #Compute rejection rates, rejection rate bars, times and labels for Cross MMD
    if not no_compute['cross_mmd']:
        rejection_rate['cross_mmd'], rejection_rate_upper['cross_mmd'], rejection_rate_lower['cross_mmd'], times['cross_mmd'], labels['cross_mmd'] = joint_group_results_dict['cross_mmd'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
    
    #Compute rejection rates, rejection rate bars, times and labels for RFF
    if not no_compute['rff']:
        rejection_rate['rff'], rejection_rate_upper['rff'], rejection_rate_lower['rff'], times['rff'], labels['rff'] = joint_group_results_dict['rff'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
    
    #Compute rejection rates, rejection rate bars, times and labels for RFF
    if not no_compute['cheap_rff']:
        input_labels = ['']*(len(args.cheap_rff_list_n_features)*len(args.cheap_rff_list_s))
        rejection_rate['cheap_rff'], rejection_rate_upper['cheap_rff'], rejection_rate_lower['cheap_rff'], times['cheap_rff'], labels['cheap_rff'] = joint_group_results_dict['cheap_rff'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple, labels=input_labels)
           
    #Plot settings    
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        mpl.rcParams['xtick.labelsize'] = label_size 
        mpl.rcParams['ytick.labelsize'] = label_size 
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5
    
    #Title settings
    if not args.no_title:
        if args.name == 'gaussians': 
            plt.title(f'Gaussian (mean separation$=${args.mean_diff}, $n_1,n_2={args.n}$, $B={args.B}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $n={args.n}$, $B={args.B}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $n={args.n}$, $B={args.B}$)')
        elif args.name == 'Higgs': 
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $n={args.n}$, $B={args.B}$)')
    
    line_number = 0
        
    #Plot complete line
    no_compute_complete = no_compute['complete']
    print(f'no_compute_complete: {no_compute_complete}')
    if not no_compute['complete']:
        print('labels[complete]', labels['complete'])
        print('plot line complete')
        label_position = (-14,0)
        plot_line(rejection_rate['complete'], rejection_rate_upper['complete'], rejection_rate_lower['complete'], times['complete'], labels['complete'], legend_text='Standard', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #Plot cheap perm line
    first_point = 0
    no_compute_cheap_perm = no_compute['cheap_perm']
    print(f'no_compute_cheap_perm: {no_compute_cheap_perm}')
    if not no_compute['cheap_perm']:
        print('labels[cheap_perm]', labels['cheap_perm'])
        print('plot line cheap_perm')
        line_number = 2
        label_position = (0,5)
        labels_cheap_perm = labels['cheap_perm'][first_point:]
        labels_cheap_perm[3] = ''
        labels_cheap_perm[4] = ''
        labels_cheap_perm[6] = ''
        labels_cheap_perm = ['$s=8$','$s=16$','$s=32$','','','$s=256$','','','$s=2048$','$s=4096$','$s=8192$','Standard']
        if args.no_point_label_cheap:
            labels_cheap_perm = ['']*len(labels_cheap_perm)
        print(f'labels_cheap_perm: {labels_cheap_perm}')
        plot_line(rejection_rate['cheap_perm'][first_point:], rejection_rate_upper['cheap_perm'][first_point:], rejection_rate_lower['cheap_perm'][first_point:], times['cheap_perm'][first_point:], labels_cheap_perm, legend_text='Cheap $n=16384$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #Plot cross MMD line
    no_compute_cheap_perm = no_compute['cross_mmd']
    print(f'no_compute_cross_mmd: {no_compute_cheap_perm}')
    if not no_compute['cross_mmd']:
        print('labels[cross_mmd]', labels['cross_mmd'])
        print('plot line cross_mmd')
        label_position = (0,0)
        plot_line(rejection_rate['cross_mmd'], rejection_rate_upper['cross_mmd'], rejection_rate_lower['cross_mmd'], times['cross_mmd'], labels['cross_mmd'], legend_text='Cross MMD', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1  
            
    #Plot RFF line
    first_point = 1
    if not no_compute['rff']:
        legend_content = 'RFF'
        label_position = (-10,9)
        plot_line(rejection_rate['rff'][first_point:], rejection_rate_upper['rff'][first_point:], rejection_rate_lower['rff'][first_point:], times['rff'][first_point:], labels['rff'][first_point:], legend_text=legend_content, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1 
    
    #Plot Cheap RFF line
    first_point = 1
    if not no_compute['cheap_rff']:
        # line_number += 1
        legend_content = 'Cheap'
        label_position = (-14,2) #(0,-9)
        num_features = len(args.cheap_rff_list_n_features)
        num_s = len(args.cheap_rff_list_s)
        print
        index_selection = num_s*np.arange(num_features)
        # for i, s in enumerate(args.cheap_rff_list_s):
        for i in range(len(args.cheap_rff_list_s) - 1, -1, -1):
            s = args.cheap_rff_list_s[i]
            if s == 16:
                continue
            print(f'i+index_selection: {i+index_selection}')
            time_seq = [times['cheap_rff'][j] for j in i+index_selection]
            rr_seq = [rejection_rate['cheap_rff'][j] for j in i+index_selection]
            rru_seq = [rejection_rate_upper['cheap_rff'][j] for j in i+index_selection]
            rrl_seq = [rejection_rate_lower['cheap_rff'][j] for j in i+index_selection]
            if i == num_s - 1:
                labels_seq = ['$r=8$','$r=32$','$r=128$','$r=512$','$r=2048$']
                legend_text = 'Standard RFF'
            else:
                labels_seq = [labels['cheap_rff'][j] for j in i+index_selection]
                legend_text = legend_content+' $s=$'+str(2*s)
            print(f'labels_seq: {labels_seq}')
            plot_line(rr_seq[first_point:], rru_seq[first_point:], rrl_seq[first_point:], time_seq[first_point:], labels_seq[first_point:], legend_text=legend_text, marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1
        
    #Plot level line
    if not args.no_nominal_level:
        plt.axhline(y=args.alpha, label='Level 0.05', color='orange', linestyle=':')

    # Define desired tick positions and labels
    if no_compute['cheap_rff']:
        tick_positions = [2, 5, 10, 20, 50, 100, 200]
        tick_labels = [2, 5, 10, 20, 50, 100, 200]
    else:
        tick_positions = [0.1, 1, 10, 100]
        tick_labels = [0.1, 1, 10, 100]

    # Set the tick positions and labels
    plt.xticks(tick_positions, tick_labels)

    plt.xlabel('Total computation time (s)')
    plt.ylabel('Power')
   
    if not no_compute['cheap_perm']:
        plt.legend(handletextpad=0.0, loc='lower right')
    else:
        plt.legend(handletextpad=0.0)
    
def fig_rejection_independence(args, test_groups, joint_group_results_dict):
    
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)
    
    #Compute rejection rates, rejection rate bars, times and labels for Complete
    if not no_compute['ind_complete_WB']:
        rejection_rate['ind_complete_WB'], rejection_rate_upper['ind_complete_WB'], rejection_rate_lower['ind_complete_WB'], times['ind_complete_WB'], labels['ind_complete_WB'] = joint_group_results_dict['ind_complete_WB'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
        
    #Compute rejection rates, rejection rate bars, times and labels for Cheap Permutations
    if not no_compute['ind_cheap_perm_WB']:
        rejection_rate['ind_cheap_perm_WB'], rejection_rate_upper['ind_cheap_perm_WB'], rejection_rate_lower['ind_cheap_perm_WB'], times['ind_cheap_perm_WB'], labels['ind_cheap_perm_WB'] = joint_group_results_dict['ind_cheap_perm_WB'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
           
    #Plot settings    
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        mpl.rcParams['xtick.labelsize'] = label_size 
        mpl.rcParams['ytick.labelsize'] = label_size 
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5
    
    #Title settings
    if not args.no_title:
        if args.name == 'gaussians_cov': 
            plt.title(f'Gaussian independence (cross covariance$=${args.cross_covariance}, $n={args.n}$, $B={args.B}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $n={args.n}$, $B={args.B}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $n={args.n}$, $B={args.B}$)')
        elif args.name == 'Higgs': 
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $n={args.n}$, $B={args.B}$)')
    
    line_number = 0
        
    #Plot complete line
    no_compute_complete = no_compute['ind_complete_WB']
    print(f'no_compute_ind_complete_WB: {no_compute_complete}')
    if not no_compute['ind_complete_WB']:
        print('labels[ind_complete_WB]', labels['ind_complete_WB'])
        print('plot line ind_complete_WB')
        label_position = (-14,0)
        plot_line(rejection_rate['ind_complete_WB'], rejection_rate_upper['ind_complete_WB'], rejection_rate_lower['ind_complete_WB'], times['ind_complete_WB'], labels['ind_complete_WB'], legend_text='Standard', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #Plot cheap perm line
    first_point = 1
    no_compute_cheap_perm = no_compute['ind_cheap_perm_WB']
    print(f'no_compute_ind_cheap_perm_WB: {no_compute_cheap_perm}')
    if not no_compute['ind_cheap_perm_WB']:
        line_number = 2
        print('labels[ind_cheap_perm_WB]', labels['ind_cheap_perm_WB'])
        print('plot line ind_cheap_perm_WB')
        label_position = (2,4)
        labels_cheap_perm = ['$s=8$','$s=16$','$s=32$','','','$s=256$','$s=512$','$s=1024$','Standard']
        plot_line(rejection_rate['ind_cheap_perm_WB'][first_point:], rejection_rate_upper['ind_cheap_perm_WB'][first_point:], rejection_rate_lower['ind_cheap_perm_WB'][first_point:], times['ind_cheap_perm_WB'][first_point:], labels_cheap_perm, legend_text='Cheap $n=2048$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #Plot level line
    if not args.no_nominal_level:
        plt.axhline(y=args.alpha, label='Level 0.05', color='orange', linestyle=':')
    
    tick_positions = [0.1, 0.2, 0.5, 1, 2]
    tick_labels = [0.1, 0.2, 0.5, 1, 2]

    # Set the tick positions and labels
    plt.xticks(tick_positions, tick_labels)

    plt.xlabel('Total computation time (s)') 
    plt.ylabel('Power')
    
    plt.legend(handletextpad=0.0)
    
def fig_B_two_sample(args, test_groups, joint_group_results_dict):
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)

    rejection_rate['complete'] = [None] * len(args.n_permutations_list)
    rejection_rate_upper['complete'] = [None] * len(args.n_permutations_list)
    rejection_rate_lower['complete'] = [None] * len(args.n_permutations_list)
    times['complete'] = [None] * len(args.n_permutations_list)
    labels['complete'] = [None] * len(args.n_permutations_list)
    
    for k, n_perm in enumerate(args.n_permutations_list):
        rejection_rate['complete'][k], rejection_rate_upper['complete'][k], rejection_rate_lower['complete'][k], times['complete'][k], labels['complete'][k] = joint_group_results_dict[str(n_perm)+'_'+'complete'].get_lists(wilson_intervals= args.wilson_intervals,z=std_multiple,labels=[str(n_perm)])
        rejection_rate['complete'][k] = rejection_rate['complete'][k][0]
        rejection_rate_upper['complete'][k] = rejection_rate_upper['complete'][k][0]
        rejection_rate_lower['complete'][k] = rejection_rate_lower['complete'][k][0]
        times['complete'][k] = times['complete'][k][0]
        labels['complete'][k] = labels['complete'][k][0]
        
    rejection_rate['complete'] = np.array(rejection_rate['complete'])
    rejection_rate_upper['complete'] = np.array(rejection_rate_upper['complete'])
    rejection_rate_lower['complete'] = np.array(rejection_rate_lower['complete'])
    times['complete'] = np.array(times['complete'])
    labels['complete'] = np.array(labels['complete'])
    
    the_rejection_rate = rejection_rate['complete']
    the_rejection_rate_upper = rejection_rate_upper['complete']
    the_rejection_rate_lower = rejection_rate_lower['complete']
    the_times = times['complete']
    the_labels = labels['complete']
    print(f'args.n_permutations_list: {args.n_permutations_list}')
    print(f'rejection_rate: {the_rejection_rate}')
    print(f'rejection_rate_upper: {the_rejection_rate_upper}')
    print(f'rejection_rate_lower: {the_rejection_rate_lower}')
    print(f'times: {the_times}')
    print(f'labels: {the_labels}')
    
    #Plot settings    
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5

    #Title settings
    if not args.no_title:
        if args.name == 'gaussians': 
            plt.title(f'Gaussian (mean separation$=${args.mean_diff}, $n={args.n}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $n={args.n}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $n={args.n}$)')
        elif args.name == 'Higgs': 
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $n={args.n}$)')
        
    line_number = 0
        
    #Plot complete line
    label_position = (-4,5)
    plot_line(rejection_rate['complete'], rejection_rate_upper['complete'], rejection_rate_lower['complete'], times['complete'], labels['complete'], legend_text='Standard', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
    
    # Define desired tick positions and labels
    tick_positions = [5, 10, 20, 50, 100]
    tick_labels = [5, 10, 20, 50, 100]

    # Set the tick positions and labels
    plt.xticks(tick_positions, tick_labels)

    plt.xlabel('Total computation time (s)')
    plt.ylabel('Power')
    
    plt.legend(handletextpad=0.0, loc='upper left')
    
def fig_B_independence(args, test_groups, joint_group_results_dict):
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)

    rejection_rate['ind_complete_WB'] = [None] * len(args.n_permutations_list)
    rejection_rate_upper['ind_complete_WB'] = [None] * len(args.n_permutations_list)
    rejection_rate_lower['ind_complete_WB'] = [None] * len(args.n_permutations_list)
    times['ind_complete_WB'] = [None] * len(args.n_permutations_list)
    labels['ind_complete_WB'] = [None] * len(args.n_permutations_list)
    
    for k, n_perm in enumerate(args.n_permutations_list):
        rejection_rate['ind_complete_WB'][k], rejection_rate_upper['ind_complete_WB'][k], rejection_rate_lower['ind_complete_WB'][k], times['ind_complete_WB'][k], labels['ind_complete_WB'][k] = joint_group_results_dict[str(n_perm)+'_'+'ind_complete_WB'].get_lists(wilson_intervals= args.wilson_intervals,z=std_multiple,labels=[str(n_perm)])
        rejection_rate['ind_complete_WB'][k] = rejection_rate['ind_complete_WB'][k][0]
        rejection_rate_upper['ind_complete_WB'][k] = rejection_rate_upper['ind_complete_WB'][k][0]
        rejection_rate_lower['ind_complete_WB'][k] = rejection_rate_lower['ind_complete_WB'][k][0]
        times['ind_complete_WB'][k] = times['ind_complete_WB'][k][0]
        labels['ind_complete_WB'][k] = labels['ind_complete_WB'][k][0]
        
    rejection_rate['ind_complete_WB'] = np.array(rejection_rate['ind_complete_WB'])
    rejection_rate_upper['ind_complete_WB'] = np.array(rejection_rate_upper['ind_complete_WB'])
    rejection_rate_lower['ind_complete_WB'] = np.array(rejection_rate_lower['ind_complete_WB'])
    times['ind_complete_WB'] = np.array(times['ind_complete_WB'])
    labels['ind_complete_WB'] = np.array(labels['ind_complete_WB'])
    
    the_rejection_rate = rejection_rate['ind_complete_WB']
    the_rejection_rate_upper = rejection_rate_upper['ind_complete_WB']
    the_rejection_rate_lower = rejection_rate_lower['ind_complete_WB']
    the_times = times['ind_complete_WB']
    the_labels = labels['ind_complete_WB']
    print(f'args.n_permutations_list: {args.n_permutations_list}')
    print(f'rejection_rate: {the_rejection_rate}')
    print(f'rejection_rate_upper: {the_rejection_rate_upper}')
    print(f'rejection_rate_lower: {the_rejection_rate_lower}')
    print(f'times: {the_times}')
    print(f'labels: {the_labels}')

    #Plot settings
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5
        
    #Title settings
    if not args.no_title:
        if args.name == 'gaussians_cov': 
            plt.title(f'Gaussian (cross covariance$=${args.cross_covariance}, $n={args.n}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $n={args.n}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $n={args.n}$)')
        elif args.name == 'Higgs': 
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $n={args.n}$)')
        
    line_number = 0
        
    #Plot complete line
    label_position = (-4,5)
    plot_line(rejection_rate['ind_complete_WB'], rejection_rate_upper['ind_complete_WB'], rejection_rate_lower['ind_complete_WB'], times['ind_complete_WB'], labels['ind_complete_WB'], legend_text='Standard', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)

    # Define desired tick positions and labels
    tick_positions = [0.1, 0.2, 0.5, 1, 2, 5]
    tick_labels = [0.1, 0.2, 0.5, 1, 2, 5]

    # Set the tick positions and labels
    plt.xticks(tick_positions, tick_labels)

    plt.xlabel('Total computation time (s)')
    plt.ylabel('Power')
    
    plt.legend(handletextpad=0.0,loc='upper left')
    
def fig_n_two_sample(args, test_groups, joint_group_results_dict, plot_time=True):
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    print(f'joint_group_results_dict.keys(): {joint_group_results_dict.keys()}')

    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)
    
    id_values_cheap_perm = 0

    test_groups_id_values = []
    if not no_compute['complete']:
        test_groups_id_values += [('complete',0)]
    if not no_compute['cheap_perm']:
        test_groups_id_values += [('cheap_perm',0),('cheap_perm',2),('cheap_perm',4),('cheap_perm',6)]
    if not no_compute['cross_mmd']:
        test_groups_id_values += [('cross_mmd',0)]

    for test_group, id_values in test_groups_id_values:
        rejection_rate[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        rejection_rate_upper[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        rejection_rate_lower[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        times[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        labels[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        
        for k, n_samples in enumerate(args.n_samples_list):
            rr, rru, rrl, ts, lb = joint_group_results_dict[str(n_samples)+'_'+test_group].get_lists(wilson_intervals= args.wilson_intervals,z=std_multiple,labels=[str(n_samples)])
            test_group_id_values = test_group+'_'+str(id_values)
            print(f'len(rr): {len(rr)}, test_group: {test_group}, id_values: {id_values}')
            rejection_rate[test_group+'_'+str(id_values)][k] = rr[id_values]
            rejection_rate_upper[test_group+'_'+str(id_values)][k] = rru[id_values]
            rejection_rate_lower[test_group+'_'+str(id_values)][k] = rrl[id_values]
            times[test_group+'_'+str(id_values)][k] = ts[id_values]
            labels[test_group+'_'+str(id_values)][k] = lb[0]

        rejection_rate[test_group+'_'+str(id_values)] = np.array(rejection_rate[test_group+'_'+str(id_values)])
        rejection_rate_upper[test_group+'_'+str(id_values)] = np.array(rejection_rate_upper[test_group+'_'+str(id_values)])
        rejection_rate_lower[test_group+'_'+str(id_values)] = np.array(rejection_rate_lower[test_group+'_'+str(id_values)])
        times[test_group+'_'+str(id_values)] = np.array(times[test_group+'_'+str(id_values)])
        labels[test_group+'_'+str(id_values)] = np.array(labels[test_group+'_'+str(id_values)])
    
        the_rejection_rate = rejection_rate[test_group+'_'+str(id_values)]
        the_rejection_rate_upper = rejection_rate_upper[test_group+'_'+str(id_values)]
        the_rejection_rate_lower = rejection_rate_lower[test_group+'_'+str(id_values)]
        the_times = times[test_group+'_'+str(id_values)]
        the_labels = labels[test_group+'_'+str(id_values)]
        print(f'Test group: {test_group}')
        print(f'args.n_permutations_list: {args.n_permutations_list}')
        print(f'rejection_rate: {the_rejection_rate}')
        print(f'rejection_rate_upper: {the_rejection_rate_upper}')
        print(f'rejection_rate_lower: {the_rejection_rate_lower}')
        print(f'times: {the_times}')
        print(f'labels: {the_labels}')

    plt.clf()
    #Plot settings
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        if plot_time:
            mpl.rcParams['xtick.labelsize'] = label_size
        else:
            mpl.rcParams['xtick.labelsize'] = 6
        mpl.rcParams['ytick.labelsize'] = label_size
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5
        
    #Title settings
    if not args.no_title:
        if args.name == 'gaussians': 
            plt.title(f'Gaussian (mean separation$=${args.mean_diff}, $B={args.B}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $B={args.B}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $B={args.B}$)')
        elif args.name == 'Higgs': 
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $B={args.B}$)')
        
    line_number = 0

    if not plot_time:
        array_n_samples = np.array(args.n_samples_list)
        
    #Plot complete line
    no_compute_complete = no_compute['complete']
    print(f'no_compute_complete: {no_compute_complete}')
    if not no_compute['complete']:
        print('labels[complete]', labels['complete_0'])
        print('plot line complete')
        label_position = (-17,0) #(-14,0)
        labels_complete = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['complete_0']]
        if plot_time:
            plot_line(rejection_rate['complete_0'], rejection_rate_upper['complete_0'], rejection_rate_lower['complete_0'], times['complete_0'], labels_complete, legend_text='Standard MMD', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        else:
            labels_complete = [''] * len(labels_complete)
            plot_line(rejection_rate['complete_0'], rejection_rate_upper['complete_0'], rejection_rate_lower['complete_0'], 2*array_n_samples, labels_complete, legend_text='Standard MMD', marker=mss[line_number], markersize=6, color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
    else:
        line_number += 1

    plot_512 = no_compute['cross_mmd']
    plot_8 = no_compute['cross_mmd']
        
    #Plot cheap perm line
    if plot_512:
        first_point = 1
        no_compute_cheap_perm = no_compute['cheap_perm']
        print(f'no_compute_cheap_perm: {no_compute_cheap_perm}')
        if not no_compute['cheap_perm']:
            print('labels[cheap_perm_6]', labels['cheap_perm_6'])
            print('plot line cheap_perm')
            label_position = (2,5)
            labels_cheap_perm = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['cheap_perm_6']]
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            print(f'labels_cheap_perm: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['cheap_perm_6'], rejection_rate_upper['cheap_perm_6'], rejection_rate_lower['cheap_perm_6'], times['cheap_perm_6'], labels_cheap_perm, legend_text='Cheap $s=512$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                labels_cheap_perm = [''] * len(labels_cheap_perm)
                plot_line(rejection_rate['cheap_perm_6'], rejection_rate_upper['cheap_perm_6'], rejection_rate_lower['cheap_perm_6'], 2*array_n_samples, labels_cheap_perm, legend_text='Cheap $s=512$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot cheap perm line
    line_number = 2
    first_point = 1
    no_compute_cheap_perm = no_compute['cheap_perm']
    print(f'no_compute_cheap_perm: {no_compute_cheap_perm}')
    if not no_compute['cheap_perm']:
        print('labels[cheap_perm_4]', labels['cheap_perm_4'])
        print('plot line cheap_perm')
        label_position = (2,5)
        labels_cheap_perm = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['cheap_perm_4']]
        labels_cheap_perm = [''] * len(labels_cheap_perm)
        print(f'labels_cheap_perm: {labels_cheap_perm}')
        if plot_time:
            plot_line(rejection_rate['cheap_perm_4'], rejection_rate_upper['cheap_perm_4'], rejection_rate_lower['cheap_perm_4'], times['cheap_perm_4'], labels_cheap_perm, legend_text='Cheap $s=128$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        else:
            plot_line(rejection_rate['cheap_perm_4'], rejection_rate_upper['cheap_perm_4'], rejection_rate_lower['cheap_perm_4'], 2*array_n_samples, labels_cheap_perm, legend_text='Cheap $s=128$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
    
    #Plot cheap perm line
    first_point = 1
    no_compute_cheap_perm = no_compute['cheap_perm']
    print(f'no_compute_cheap_perm: {no_compute_cheap_perm}')
    if not no_compute['cheap_perm']:
        print('labels[cheap_perm_2]', labels['cheap_perm_2'])
        print('plot line cheap_perm')
        label_position = (-15.5,3) if no_compute['cross_mmd'] else (22.5,-1)
        labels_cheap_perm = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['cheap_perm_2']]
        print(f'labels_cheap_perm: {labels_cheap_perm}')
        if plot_time:
            plot_line(rejection_rate['cheap_perm_2'], rejection_rate_upper['cheap_perm_2'], rejection_rate_lower['cheap_perm_2'], times['cheap_perm_2'], labels_cheap_perm, legend_text='Cheap $s=32$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        else:
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            plot_line(rejection_rate['cheap_perm_2'], rejection_rate_upper['cheap_perm_2'], rejection_rate_lower['cheap_perm_2'], 2*array_n_samples, labels_cheap_perm, legend_text='Cheap $s=32$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1

    #Plot cheap perm line
    if plot_8:
        first_point = 1
        no_compute_cheap_perm = no_compute['cheap_perm']
        print(f'no_compute_cheap_perm: {no_compute_cheap_perm}')
        if not no_compute['cheap_perm']:
            print('labels[cheap_perm_0]', labels['cheap_perm_0'])
            print('plot line cheap_perm')
            label_position = (2,2)
            labels_cheap_perm = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['cheap_perm_0']]
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            print(f'labels_cheap_perm: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['cheap_perm_0'], rejection_rate_upper['cheap_perm_0'], rejection_rate_lower['cheap_perm_0'], times['cheap_perm_0'], labels_cheap_perm, legend_text='Cheap $s=8$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['cheap_perm_0'], rejection_rate_upper['cheap_perm_0'], rejection_rate_lower['cheap_perm_0'], 2*array_n_samples, labels_cheap_perm, legend_text='Cheap $s=8$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot cheap perm line
    # first_point = 1
    # no_compute_cheap_perm = no_compute['cheap_perm']
    # print(f'no_compute_cheap_perm: {no_compute_cheap_perm}')
    # if not no_compute['cheap_perm']:
    #     print('labels[cheap_perm_1]', labels['cheap_perm_1'])
    #     print('plot line cheap_perm')
    #     label_position = (2,2)
    #     labels_cheap_perm = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['cheap_perm_1']]
    #     labels_cheap_perm = [''] * len(labels_cheap_perm)
    #     print(f'labels_cheap_perm: {labels_cheap_perm}')
    #     if plot_time:
    #         plot_line(rejection_rate['cheap_perm_1'], rejection_rate_upper['cheap_perm_1'], rejection_rate_lower['cheap_perm_1'], times['cheap_perm_1'], labels_cheap_perm, legend_text='Cheap $s=16$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
    #     else:
    #         plot_line(rejection_rate['cheap_perm_1'], rejection_rate_upper['cheap_perm_1'], rejection_rate_lower['cheap_perm_1'], 2*array_n_samples, labels_cheap_perm, legend_text='Cheap $s=16$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
    #     line_number += 1

    #Plot cross MMD line
    line_number = 6
    no_compute_cheap_perm = no_compute['cross_mmd']
    print(f'no_compute_cross_mmd: {no_compute_cheap_perm}')
    if not no_compute['cross_mmd']:
        print('labels[cross_mmd_0]', labels['cross_mmd_0'])
        print('plot line cross_mmd')
        label_position = (-14,2)
        labels_cross_mmd = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['cross_mmd_0']]
        if plot_time:
            plot_line(rejection_rate['cross_mmd_0'], rejection_rate_upper['cross_mmd_0'], rejection_rate_lower['cross_mmd_0'], times['cross_mmd_0'], labels_cross_mmd, legend_text='Asymp. cross-MMD', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        else:
            labels_cross_mmd = [''] * len(labels_cross_mmd)
            plot_line(rejection_rate['cross_mmd_0'], rejection_rate_upper['cross_mmd_0'], rejection_rate_lower['cross_mmd_0'], 2*array_n_samples, labels_cross_mmd, legend_text='Asymp. cross-MMD', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1  
    
    if plot_time:
        # Define desired tick positions and labels
        if not no_compute['cross_mmd']:
            tick_positions = [0.01, 0.1, 1, 10, 100, 1000]
            tick_labels = [0.01, 0.1, 1, 10, 100, 1000]
        else:
            tick_positions = [0.1, 1, 10, 100, 1000]
            tick_labels = [0.1, 1, 10, 100, 1000]

        # Set the tick positions and labels
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Total computation time (s)')
    else:
        # Remove existing tick labels
        plt.tick_params(axis='x', which='minor', labelbottom=False)
        # Define desired tick positions and labels
        tick_positions = [2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
        tick_labels = [2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
        # Set the tick positions and labels
        plt.xticks(tick_positions, tick_labels)
        # Set the tick positions and labels
        plt.xlabel('Sample size $n$')
    plt.ylabel('Power')

    if plot_time:
        if not no_compute['complete'] and no_compute['cross_mmd']:
            plt.xlim([0.01,1500])
        elif no_compute['complete'] and not no_compute['cross_mmd']:
            plt.xlim([0.1,33])
        else:
            plt.xlim([0.0029,1500])
    else:
        plt.xlim([1536,40960])

    plt.legend(handletextpad=0.0, loc='upper left')
    
def fig_n_independence(args, test_groups, joint_group_results_dict, plot_time=True):
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)
    
    print(f'joint_group_results_dict.keys(): {joint_group_results_dict.keys()}')
    
    #id_values_cheap_perm = 5
    test_groups_id_values = []
    if not no_compute['ind_complete_WB']:
        test_groups_id_values += [('ind_complete_WB',0)]
    if not no_compute['ind_cheap_perm_WB']:
        test_groups_id_values += [('ind_cheap_perm_WB',0),('ind_cheap_perm_WB',1),('ind_cheap_perm_WB',2),('ind_cheap_perm_WB',4),('ind_cheap_perm_WB',5),('ind_cheap_perm_WB',6)]
    if not no_compute['ind_cross']:
        test_groups_id_values += [('ind_cross',0)]

    #for test_group, id_values in [('ind_complete_WB',0),('ind_cheap_perm_WB',1),('ind_cheap_perm_WB',2),('ind_cheap_perm_WB',3),('ind_cross',0)]:
    for test_group, id_values in test_groups_id_values:    
        rejection_rate[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        rejection_rate_upper[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        rejection_rate_lower[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        times[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        labels[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list)
        
        for k, n_samples in enumerate(args.n_samples_list):
            rr, rru, rrl, ts, lb = joint_group_results_dict[str(n_samples)+'_'+test_group].get_lists(wilson_intervals= args.wilson_intervals,z=std_multiple,labels=[str(n_samples)])
            rejection_rate[test_group+'_'+str(id_values)][k] = rr[id_values]
            rejection_rate_upper[test_group+'_'+str(id_values)][k] = rru[id_values]
            rejection_rate_lower[test_group+'_'+str(id_values)][k] = rrl[id_values]
            times[test_group+'_'+str(id_values)][k] = ts[id_values]
            labels[test_group+'_'+str(id_values)][k] = lb[0]

        rejection_rate[test_group+'_'+str(id_values)] = np.array(rejection_rate[test_group+'_'+str(id_values)])
        rejection_rate_upper[test_group+'_'+str(id_values)] = np.array(rejection_rate_upper[test_group+'_'+str(id_values)])
        rejection_rate_lower[test_group+'_'+str(id_values)] = np.array(rejection_rate_lower[test_group+'_'+str(id_values)])
        times[test_group+'_'+str(id_values)] = np.array(times[test_group+'_'+str(id_values)])
        labels[test_group+'_'+str(id_values)] = np.array(labels[test_group+'_'+str(id_values)])
    
        the_rejection_rate = rejection_rate[test_group+'_'+str(id_values)]
        the_rejection_rate_upper = rejection_rate_upper[test_group+'_'+str(id_values)]
        the_rejection_rate_lower = rejection_rate_lower[test_group+'_'+str(id_values)]
        the_times = times[test_group+'_'+str(id_values)]
        the_labels = labels[test_group+'_'+str(id_values)]
        print(f'Test group: {test_group}')
        print(f'args.n_permutations_list: {args.n_permutations_list}')
        print(f'rejection_rate: {the_rejection_rate}')
        print(f'rejection_rate_upper: {the_rejection_rate_upper}')
        print(f'rejection_rate_lower: {the_rejection_rate_lower}')
        print(f'times: {the_times}')
        print(f'labels: {the_labels}')

    #Plot settings
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        if not no_compute['ind_cross'] and plot_time: 
            mpl.rcParams['xtick.labelsize'] = 4.5
        else:
            mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5
        
    #Title settings
    if not args.no_title:
        if args.name == 'gaussians_cov': 
            plt.title(f'Gaussian (cross covariance$=${args.cross_covariance}, $B={args.B}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $B={args.B}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $B={args.B}$)')
        elif args.name == 'Higgs': 
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $B={args.B}$)')
        
    line_number = 0

    if not plot_time:
        array_n_samples = np.array(args.n_samples_list)
        
    #Plot complete line
    no_compute_ind_complete_WB = no_compute['ind_complete_WB']
    print(f'no_compute_ind_complete_WB: {no_compute_ind_complete_WB}')
    if not no_compute['ind_complete_WB']:
        print('labels[ind_complete_WB]', labels['ind_complete_WB_0'])
        print('plot line ind_complete_WB')
        label_position = (-16,0)
        labels_complete = ['$n=$'+n_samples for n_samples in labels['ind_complete_WB_0']]
        if plot_time:
            plot_line(rejection_rate['ind_complete_WB_0'], rejection_rate_upper['ind_complete_WB_0'], rejection_rate_lower['ind_complete_WB_0'], times['ind_complete_WB_0'], labels_complete, legend_text='Standard HSIC', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        else:
            labels_complete = [''] * len(labels_complete)
            plot_line(rejection_rate['ind_complete_WB_0'], rejection_rate_upper['ind_complete_WB_0'], rejection_rate_lower['ind_complete_WB_0'], array_n_samples, labels_complete, legend_text='Standard HSIC', marker=mss[line_number], markersize=6, color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
    else:
        line_number += 1

    plot_256 = no_compute['ind_cross']
    plot_128 = True
    plot_64 = True
    plot_16 = no_compute['ind_cross']
    plot_8 = False
    plot_4 = False

    #Plot cheap perm line
    if plot_256:
        first_point = 1
        no_compute_cheap_perm = no_compute['ind_cheap_perm_WB']
        print(f'no_compute_ind_cheap_perm_WB: {no_compute_cheap_perm}')
        if not no_compute['ind_cheap_perm_WB']:
            print('labels[ind_cheap_perm_WB_6]', labels['ind_cheap_perm_WB_6'])
            print('plot line ind_cheap_perm_WB')
            label_position = (2,5)
            labels_cheap_perm = ['$n=$'+n_samples for n_samples in labels['ind_cheap_perm_WB_6']]
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            print(f'labels_ind_cheap_perm_WB: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['ind_cheap_perm_WB_6'], rejection_rate_upper['ind_cheap_perm_WB_6'], rejection_rate_lower['ind_cheap_perm_WB_6'], times['ind_cheap_perm_WB_6'], labels_cheap_perm, legend_text='Cheap '+f'$s=256$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['ind_cheap_perm_WB_6'], rejection_rate_upper['ind_cheap_perm_WB_6'], rejection_rate_lower['ind_cheap_perm_WB_6'], array_n_samples, labels_cheap_perm, legend_text='Cheap '+f'$s=256$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot cheap perm line
    if plot_128:
        first_point = 1
        line_number = 2
        no_compute_cheap_perm = no_compute['ind_cheap_perm_WB']
        print(f'no_compute_ind_cheap_perm_WB: {no_compute_cheap_perm}')
        if not no_compute['ind_cheap_perm_WB']:
            print('labels[ind_cheap_perm_WB_5]', labels['ind_cheap_perm_WB_5'])
            print('plot line ind_cheap_perm_WB')
            label_position = (2,5)
            labels_cheap_perm = ['$n=$'+n_samples for n_samples in labels['ind_cheap_perm_WB_5']]
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            print(f'labels_ind_cheap_perm_WB: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['ind_cheap_perm_WB_5'], rejection_rate_upper['ind_cheap_perm_WB_5'], rejection_rate_lower['ind_cheap_perm_WB_5'], times['ind_cheap_perm_WB_5'], labels_cheap_perm, legend_text='Cheap '+f'$s=128$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['ind_cheap_perm_WB_5'], rejection_rate_upper['ind_cheap_perm_WB_5'], rejection_rate_lower['ind_cheap_perm_WB_5'], array_n_samples, labels_cheap_perm, legend_text='Cheap '+f'$s=128$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot cheap perm line
    if plot_64:
        first_point = 1
        no_compute_cheap_perm = no_compute['ind_cheap_perm_WB']
        print(f'no_compute_ind_cheap_perm_WB: {no_compute_cheap_perm}')
        if not no_compute['ind_cheap_perm_WB']:
            print('labels[ind_cheap_perm_WB_4]', labels['ind_cheap_perm_WB_4'])
            print('plot line ind_cheap_perm_WB')
            label_position = (2,5)
            labels_cheap_perm = ['$n=$'+n_samples for n_samples in labels['ind_cheap_perm_WB_4']]
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            print(f'labels_ind_cheap_perm_WB: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['ind_cheap_perm_WB_4'], rejection_rate_upper['ind_cheap_perm_WB_4'], rejection_rate_lower['ind_cheap_perm_WB_4'], times['ind_cheap_perm_WB_4'], labels_cheap_perm, legend_text='Cheap '+f'$s=64$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['ind_cheap_perm_WB_4'], rejection_rate_upper['ind_cheap_perm_WB_4'], rejection_rate_lower['ind_cheap_perm_WB_4'], array_n_samples, labels_cheap_perm, legend_text='Cheap '+f'$s=64$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot cheap perm line
    if plot_16:
        first_point = 1
        no_compute_cheap_perm = no_compute['ind_cheap_perm_WB']
        print(f'no_compute_ind_cheap_perm_WB: {no_compute_cheap_perm}')
        if not no_compute['ind_cheap_perm_WB']:
            print('labels[ind_cheap_perm_WB_2]', labels['ind_cheap_perm_WB_2'])
            print('plot line ind_cheap_perm_WB')
            label_position = (2,5)
            labels_cheap_perm = ['$n=$'+n_samples for n_samples in labels['ind_cheap_perm_WB_2']]
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            print(f'labels_ind_cheap_perm_WB: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['ind_cheap_perm_WB_2'], rejection_rate_upper['ind_cheap_perm_WB_2'], rejection_rate_lower['ind_cheap_perm_WB_2'], times['ind_cheap_perm_WB_2'], labels_cheap_perm, legend_text='Cheap '+f'$s=16$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                labels_cheap_perm = [''] * len(labels_cheap_perm)
                plot_line(rejection_rate['ind_cheap_perm_WB_2'], rejection_rate_upper['ind_cheap_perm_WB_2'], rejection_rate_lower['ind_cheap_perm_WB_2'], array_n_samples, labels_cheap_perm, legend_text='Cheap '+f'$s=16$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    if plot_8:
        #Plot cheap perm line
        first_point = 1
        no_compute_cheap_perm = no_compute['ind_cheap_perm_WB']
        print(f'no_compute_ind_cheap_perm_WB: {no_compute_cheap_perm}')
        if not no_compute['ind_cheap_perm_WB']:
            print('labels[ind_cheap_perm_WB_1]', labels['ind_cheap_perm_WB_1'])
            print('plot line ind_cheap_perm_WB')
            label_position = (2,5)
            labels_cheap_perm = ['$n=$'+n_samples for n_samples in labels['ind_cheap_perm_WB_1']]
            labels_cheap_perm = [''] * len(labels_cheap_perm)
            print(f'labels_ind_cheap_perm_WB: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['ind_cheap_perm_WB_1'], rejection_rate_upper['ind_cheap_perm_WB_1'], rejection_rate_lower['ind_cheap_perm_WB_1'], times['ind_cheap_perm_WB_1'], labels_cheap_perm, legend_text='Cheap '+f'$s=8$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['ind_cheap_perm_WB_1'], rejection_rate_upper['ind_cheap_perm_WB_1'], rejection_rate_lower['ind_cheap_perm_WB_1'], array_n_samples, labels_cheap_perm, legend_text='Cheap '+f'$s=8$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    if plot_4:
        #Plot cheap perm line
        first_point = 1
        no_compute_cheap_perm = no_compute['ind_cheap_perm_WB']
        print(f'no_compute_ind_cheap_perm_WB: {no_compute_cheap_perm}')
        if not no_compute['ind_cheap_perm_WB']:
            print('labels[ind_cheap_perm_WB_0]', labels['ind_cheap_perm_WB_0'])
            print('plot line ind_cheap_perm_WB')
            label_position = (-9,2)
            labels_cheap_perm = ['$n=$'+n_samples for n_samples in labels['ind_cheap_perm_WB_0']]
            print(f'labels_ind_cheap_perm_WB: {labels_cheap_perm}')
            if plot_time:
                plot_line(rejection_rate['ind_cheap_perm_WB_0'], rejection_rate_upper['ind_cheap_perm_WB_0'], rejection_rate_lower['ind_cheap_perm_WB_0'], times['ind_cheap_perm_WB_0'], labels_cheap_perm, legend_text='Cheap '+f'$s=4$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                labels_cheap_perm = [''] * len(labels_cheap_perm)
                plot_line(rejection_rate['ind_cheap_perm_WB_0'], rejection_rate_upper['ind_cheap_perm_WB_0'], rejection_rate_lower['ind_cheap_perm_WB_0'], array_n_samples, labels_cheap_perm, legend_text='Cheap '+f'$s=4$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1

    #Plot cross MMD line
    line_number = 6
    no_compute_cross = no_compute['ind_cross']
    print(f'no_compute_ind_cross: {no_compute_cross}')
    if not no_compute['ind_cross']:
        print('labels[ind_cross]', labels['ind_cross_0'])
        print('plot line ind_cross')
        label_position = (-14,0)
        labels_cross_mmd = ['$n=$'+n_samples for n_samples in labels['ind_cross_0']]
        if plot_time:
            plot_line(rejection_rate['ind_cross_0'], rejection_rate_upper['ind_cross_0'], rejection_rate_lower['ind_cross_0'], times['ind_cross_0'], labels_cross_mmd, legend_text='Asymp. cross-HSIC', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        else:
            labels_cross_mmd = [''] * len(labels_cross_mmd)
            plot_line(rejection_rate['ind_cross_0'], rejection_rate_upper['ind_cross_0'], rejection_rate_lower['ind_cross_0'], array_n_samples, labels_cross_mmd, legend_text='Asymp. cross-HSIC', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1  
    
    if plot_time:
        # Define desired tick positions and labels
        if not no_compute['ind_cross']:
            tick_positions = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
            tick_labels = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        else:
            tick_positions = [0.01, 0.1, 1, 10, 100]
            tick_labels = [0.01, 0.1, 1, 10, 100]

        # Set the tick positions and labels
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Total computation time (s)')
    else:
        # Define desired tick positions and labels
        tick_positions = [512, 1024, 1536, 2048, 3072, 4096]
        tick_labels = [512, 1024, 1536, 2048, 3072, 4096]

        # Set the tick positions and labels
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Sample size $n$')
    plt.ylabel('Power')
    
    if plot_time:
        if no_compute['ind_cross'] and not no_compute['ind_complete_WB']:
            plt.xlim([0.003,15])
        elif not no_compute['ind_cross'] and no_compute['ind_complete_WB']:
            plt.xlim([0.0016,2])
        else:
            plt.xlim([0.0007,13])
        plt.ylim([0.07,1.02])
    else:
        plt.xlim([384,5120])
    
    plt.legend(handletextpad=0.0, loc='upper left')
    
def fig_Wilcoxon_two_sample(args, test_groups, joint_group_results_dict):
    
    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()
    
    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2 
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2
    
    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)
        
    #Compute rejection rates, rejection rate bars, times and labels for Cheap Permutations
    if not no_compute['wilcoxon']:
        rejection_rate['wilcoxon'], rejection_rate_upper['wilcoxon'], rejection_rate_lower['wilcoxon'], times['wilcoxon'], labels['wilcoxon'] = joint_group_results_dict['wilcoxon'].get_lists(wilson_intervals=args.wilson_intervals,z=std_multiple)
           
    #Plot settings    
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        mpl.rcParams['xtick.labelsize'] = label_size 
        mpl.rcParams['ytick.labelsize'] = label_size 
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5
    
    #Title settings
    if not args.no_title:
        if args.name == 'gaussians': 
            plt.title(f'Gaussian (mean separation$=${args.mean_diff}, $n_1,n_2={args.n}$, $B={args.B}$, $d={args.d}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $n={args.n}$, $B={args.B}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $n={args.n}$, $B={args.B}$)')
        elif args.name == 'Higgs': 
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $n={args.n}$, $B={args.B}$)')
    
    line_number = 0
        
    #Plot cheap perm line
    first_point = 0
    no_compute_wilcoxon = no_compute['wilcoxon']
    print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
    if not no_compute['wilcoxon']:
        print('labels[wilcoxon]', labels['wilcoxon'])
        print('plot line wilcoxon')
        line_number = 2
        label_position = (0,4)
        labels_wilcoxon = labels['wilcoxon'][first_point:]
        labels_wilcoxon[3] = ''
        labels_wilcoxon[4] = ''
        labels_wilcoxon[5] = ''
        labels_wilcoxon[6] = ''
        labels_wilcoxon[7] = ''
        labels_wilcoxon[9] = ''
        labels_wilcoxon = ['$s=8$','$s=16$','$s=32$','','','$s=256$','','$s=1024$','$s=2048$','$s=4096$','$s=8192$','Standard']
        if args.no_point_label_cheap:
            labels_wilcoxon = ['']*len(labels_wilcoxon)
        print(f'labels_wilcoxon: {labels_wilcoxon}')
        plot_line(rejection_rate['wilcoxon'][first_point:], rejection_rate_upper['wilcoxon'][first_point:], rejection_rate_lower['wilcoxon'][first_point:], times['wilcoxon'][first_point:], labels_wilcoxon, legend_text='Cheap $n=16384$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1
        
    #Plot level line
    if not args.no_nominal_level:
        plt.axhline(y=args.alpha, label='Level 0.05', color='orange', linestyle=':')
    
    # Define desired tick positions and labels
    tick_positions = [1, 10, 100]
    tick_labels = [1, 10, 100]

    # Set the tick positions and labels
    plt.xticks(tick_positions, tick_labels)

    plt.ylim([0.45,0.625])

    if args.name == 'gaussians': 
        plt.xlabel('Total computation time (s)')
    elif args.name == 'blobs': 
        plt.xlabel('Total computation time (s)')
    elif args.name == 'EMNIST': 
        plt.xlabel('Total computation time (s)')
    elif args.name == 'Higgs': 
        plt.xlabel('Total computation time (s)')
    
    plt.ylabel('Power')
    
    plt.legend(handletextpad=0.0, loc='lower right')

def fig_n_Wilcoxon(args, test_groups, joint_group_results_dict, plot_time=True):

    rejection_rate = dict()
    rejection_rate_upper = dict()
    rejection_rate_lower = dict()
    times = dict()
    labels = dict()

    #linestyle, markers, marker sizes and colors
    lss = ['-', '-.',  ':', '--',  '--', '-.', ':', '-', '--', '-.', ':', '-']*2
    mss = ['>','s', 'o', 'D', '+', '*', 'x', '>', '<', '^', 'v']*2
    ms_size = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]*2
    colors = ['#e41a1c', 'cyan', '#0000cd', '#4daf4a', 'magenta', 'gray' ,'orange','yellow', 'black']*2

    std_multiple = scipy.stats.norm.ppf((1+args.wilson_size)/2)

    print(f'joint_group_results_dict.keys(): {joint_group_results_dict.keys()}')

    test_groups_id_values = [('wilcoxon',0),('wilcoxon',1),('wilcoxon',2),('wilcoxon',3),('wilcoxon',4),('wilcoxon',5),('wilcoxon',6),('wilcoxon',7),('wilcoxon',8)]

    for test_group, id_values in test_groups_id_values:
        rejection_rate[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list_wilcoxon)
        rejection_rate_upper[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list_wilcoxon)
        rejection_rate_lower[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list_wilcoxon)
        times[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list_wilcoxon)
        labels[test_group+'_'+str(id_values)] = [None] * len(args.n_samples_list_wilcoxon)

        for k, n_samples in enumerate(args.n_samples_list_wilcoxon):
            rr, rru, rrl, ts, lb = joint_group_results_dict[str(n_samples)+'_'+test_group].get_lists(wilson_intervals= args.wilson_intervals,z=std_multiple,labels=[str(n_samples)])
            test_group_id_values = test_group+'_'+str(id_values)
            print(f'len(rr): {len(rr)}, test_group: {test_group}, id_values: {id_values}')
            rejection_rate[test_group+'_'+str(id_values)][k] = rr[id_values]
            rejection_rate_upper[test_group+'_'+str(id_values)][k] = rru[id_values]
            rejection_rate_lower[test_group+'_'+str(id_values)][k] = rrl[id_values]
            times[test_group+'_'+str(id_values)][k] = ts[id_values]
            labels[test_group+'_'+str(id_values)][k] = lb[0]

        rejection_rate[test_group+'_'+str(id_values)] = np.array(rejection_rate[test_group+'_'+str(id_values)])
        rejection_rate_upper[test_group+'_'+str(id_values)] = np.array(rejection_rate_upper[test_group+'_'+str(id_values)])
        rejection_rate_lower[test_group+'_'+str(id_values)] = np.array(rejection_rate_lower[test_group+'_'+str(id_values)])
        times[test_group+'_'+str(id_values)] = np.array(times[test_group+'_'+str(id_values)])
        labels[test_group+'_'+str(id_values)] = np.array(labels[test_group+'_'+str(id_values)])

        the_rejection_rate = rejection_rate[test_group+'_'+str(id_values)]
        the_rejection_rate_upper = rejection_rate_upper[test_group+'_'+str(id_values)]
        the_rejection_rate_lower = rejection_rate_lower[test_group+'_'+str(id_values)]
        the_times = times[test_group+'_'+str(id_values)]
        the_labels = labels[test_group+'_'+str(id_values)]
        print(f'Test group: {test_group}')
        print(f'args.n_permutations_list: {args.n_permutations_list}')
        print(f'rejection_rate: {the_rejection_rate}')
        print(f'rejection_rate_upper: {the_rejection_rate_upper}')
        print(f'rejection_rate_lower: {the_rejection_rate_lower}')
        print(f'times: {the_times}')
        print(f'labels: {the_labels}')

    #Plot settings
    title_size = 20
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        label_size = 9
        legend_size = 8
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        mpl.rcParams['figure.titlesize'] = label_size
        mpl.rcParams['lines.markersize'] = label_size
        mpl.rcParams['grid.linewidth'] = 1.5
        mpl.rcParams['legend.fontsize'] = legend_size
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        pylab.rcParams['xtick.major.pad'] = 5
        pylab.rcParams['ytick.major.pad'] = 5

    #Title settings
    if not args.no_title:
        if args.name == 'gaussians':
            plt.title(f'Gaussian (mean separation$=${args.mean_diff}, $B={args.B}$)')
        elif args.name == 'blobs':
            plt.title(f'Blobs ($\epsilon=${args.epsilon}, $B={args.B}$)')
        elif args.name == 'EMNIST':
            plt.title(f'Downsampled EMNIST ('+r'$p_{even}=$'+f'{args.p_even}, $B={args.B}$)')
        elif args.name == 'Higgs':
            plt.title(r'Higgs ($p_{p}=$'+f'${args.p_poisoning}$'+f', $B={args.B}$)')

    line_number = 0

    if not plot_time:
        array_n_samples_wilcoxon = np.array(args.n_samples_list_wilcoxon)

    #Plot wilcoxon complete line
    first_point = 1
    no_compute_wilcoxon = no_compute['wilcoxon']
    print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
    if not no_compute['wilcoxon']:
        print('labels[wilcoxon_8]', labels['wilcoxon_8'])
        print('plot line wilcoxon complete')
        label_position = (-14,5)
        labels_wilcoxon = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['wilcoxon_8']]
        print(f'labels_wilcoxon: {labels_wilcoxon}')
        if plot_time:
            plot_line(rejection_rate['wilcoxon_8'], rejection_rate_upper['wilcoxon_8'], rejection_rate_lower['wilcoxon_8'], times['wilcoxon_8'], labels_wilcoxon, legend_text='Standard WMW', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        else:
            labels_wilcoxon = [''] * len(labels_wilcoxon)
            plot_line(rejection_rate['wilcoxon_8'], rejection_rate_upper['wilcoxon_8'], rejection_rate_lower['wilcoxon_8'], 2*array_n_samples, labels_wilcoxon, legend_text='Standard WMW', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
        line_number += 1

    plot_512 = False
    plot_256 = True
    plot_128 = True
    plot_64 = True
    plot_32 = False
    plot_16 = True
    plot_8 = False

    #Plot wilcoxon line
    if plot_512:
        first_point = 1
        no_compute_wilcoxon = no_compute['wilcoxon']
        print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
        if not no_compute['wilcoxon']:
            print('labels[wilcoxon_6]', labels['wilcoxon_6'])
            print('plot line wilcoxon')
            label_position = (2,5)
            labels_wilcoxon = ['n='+str(2*int(n_samples)) for n_samples in labels['wilcoxon_6']]
            labels_wilcoxon = [''] * len(labels_wilcoxon)
            print(f'labels_wilcoxon: {labels_wilcoxon}')
            if plot_time:
                plot_line(rejection_rate['wilcoxon_6'], rejection_rate_upper['wilcoxon_6'], rejection_rate_lower['wilcoxon_6'], times['wilcoxon_6'], labels_wilcoxon, legend_text='Cheap $s=512$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                labels_wilcoxon = [''] * len(labels_wilcoxon)
                plot_line(rejection_rate['wilcoxon_6'], rejection_rate_upper['wilcoxon_6'], rejection_rate_lower['wilcoxon_6'], 2*array_n_samples, labels_wilcoxon, legend_text='Cheap $s=512$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot wilcoxon line
    if plot_256:
        first_point = 1
        no_compute_wilcoxon = no_compute['wilcoxon']
        print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
        if not no_compute['wilcoxon']:
            print('labels[wilcoxon_5]', labels['wilcoxon_5'])
            print('plot line wilcoxon')
            label_position = (2,5)
            labels_wilcoxon = ['n='+str(2*int(n_samples)) for n_samples in labels['wilcoxon_5']]
            labels_wilcoxon = [''] * len(labels_wilcoxon)
            print(f'labels_wilcoxon: {labels_wilcoxon}')
            if plot_time:
                plot_line(rejection_rate['wilcoxon_5'], rejection_rate_upper['wilcoxon_5'], rejection_rate_lower['wilcoxon_5'], times['wilcoxon_5'], labels_wilcoxon, legend_text='Cheap $s=256$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['wilcoxon_5'], rejection_rate_upper['wilcoxon_5'], rejection_rate_lower['wilcoxon_5'], 2*array_n_samples, labels_wilcoxon, legend_text='Cheap $s=256$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot wilcoxon line
    if plot_128:
        first_point = 1
        no_compute_wilcoxon = no_compute['wilcoxon']
        print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
        if not no_compute['wilcoxon']:
            print('labels[wilcoxon_4]', labels['wilcoxon_4'])
            print('plot line wilcoxon')
            label_position = (-14,6)
            labels_wilcoxon = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['wilcoxon_4']]
            labels_wilcoxon = [''] * len(labels_wilcoxon)
            print(f'labels_wilcoxon: {labels_wilcoxon}')
            if plot_time:
                plot_line(rejection_rate['wilcoxon_4'], rejection_rate_upper['wilcoxon_4'], rejection_rate_lower['wilcoxon_4'], times['wilcoxon_4'], labels_wilcoxon, legend_text='Cheap $s=128$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['wilcoxon_4'], rejection_rate_upper['wilcoxon_4'], rejection_rate_lower['wilcoxon_4'], 2*array_n_samples, labels_wilcoxon, legend_text='Cheap $s=128$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot wilcoxon line
    if plot_64:
        first_point = 1
        no_compute_wilcoxon = no_compute['wilcoxon']
        print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
        if not no_compute['wilcoxon']:
            print('labels[wilcoxon_3]', labels['wilcoxon_3'])
            print('plot line wilcoxon')
            label_position = (2,2)
            labels_wilcoxon = ['n='+str(2*int(n_samples)) for n_samples in labels['wilcoxon_3']]
            labels_wilcoxon = [''] * len(labels_wilcoxon)
            print(f'labels_wilcoxon: {labels_wilcoxon}')
            if plot_time:
                plot_line(rejection_rate['wilcoxon_3'], rejection_rate_upper['wilcoxon_3'], rejection_rate_lower['wilcoxon_3'], times['wilcoxon_3'], labels_wilcoxon, legend_text='Cheap $s=64$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['wilcoxon_3'], rejection_rate_upper['wilcoxon_3'], rejection_rate_lower['wilcoxon_3'], 2*array_n_samples, labels_wilcoxon, legend_text='Cheap $s=64$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot wilcoxon line
    if plot_32:
        first_point = 1
        no_compute_wilcoxon = no_compute['wilcoxon']
        print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
        if not no_compute['wilcoxon']:
            print('labels[wilcoxon_2]', labels['wilcoxon_2'])
            print('plot line wilcoxon')
            label_position = (-17,3)
            labels_wilcoxon = ['$n=$'+str(2*int(n_samples)) for n_samples in labels['wilcoxon_2']]
            print(f'labels_wilcoxon: {labels_wilcoxon}')
            if plot_time:
                plot_line(rejection_rate['wilcoxon_2'], rejection_rate_upper['wilcoxon_2'], rejection_rate_lower['wilcoxon_2'], times['wilcoxon_2'], labels_wilcoxon, legend_text='Cheap $s=32$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                labels_wilcoxon = [''] * len(labels_wilcoxon)
                plot_line(rejection_rate['wilcoxon_2'], rejection_rate_upper['wilcoxon_2'], rejection_rate_lower['wilcoxon_2'], 2*array_n_samples, labels_wilcoxon, legend_text='Cheap $s=32$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot wilcoxon line
    if plot_16:
        first_point = 1
        no_compute_wilcoxon = no_compute['wilcoxon']
        print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
        if not no_compute['wilcoxon']:
            print('labels[wilcoxon_1]', labels['wilcoxon_1'])
            print('plot line wilcoxon')
            label_position = (2,2)
            labels_wilcoxon = ['n='+str(2*int(n_samples)) for n_samples in labels['wilcoxon_1']]
            labels_wilcoxon = [''] * len(labels_wilcoxon)
            print(f'labels_wilcoxon: {labels_wilcoxon}')
            if plot_time:
                plot_line(rejection_rate['wilcoxon_1'], rejection_rate_upper['wilcoxon_1'], rejection_rate_lower['wilcoxon_1'], times['wilcoxon_1'], labels_wilcoxon, legend_text='Cheap $s=16$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['wilcoxon_1'], rejection_rate_upper['wilcoxon_1'], rejection_rate_lower['wilcoxon_1'], 2*array_n_samples, labels_wilcoxon, legend_text='Cheap $s=16$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    #Plot wilcoxon line
    if plot_8:
        first_point = 1
        no_compute_wilcoxon = no_compute['wilcoxon']
        print(f'no_compute_wilcoxon: {no_compute_wilcoxon}')
        if not no_compute['wilcoxon']:
            print('labels[wilcoxon_0]', labels['wilcoxon_0'])
            print('plot line wilcoxon')
            label_position = (2,2)
            labels_wilcoxon = ['n='+str(2*int(n_samples)) for n_samples in labels['wilcoxon_0']]
            labels_wilcoxon = [''] * len(labels_wilcoxon)
            print(f'labels_wilcoxon: {labels_wilcoxon}')
            if plot_time:
                plot_line(rejection_rate['wilcoxon_0'], rejection_rate_upper['wilcoxon_0'], rejection_rate_lower['wilcoxon_0'], times['wilcoxon_0'], labels_wilcoxon, legend_text='Cheap $s=8$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            else:
                plot_line(rejection_rate['wilcoxon_0'], rejection_rate_upper['wilcoxon_0'], rejection_rate_lower['wilcoxon_0'], 2*array_n_samples, labels_wilcoxon, legend_text='Cheap $s=8$', marker=mss[line_number], markersize=ms_size[line_number], color=colors[line_number], linestyle=lss[line_number], xytext=label_position, log_time_scale=args.log_time_scale, small_times=args.small_times)
            line_number += 1

    if plot_time:
        # Define desired tick positions and labels
        tick_positions = [0.001, 0.01, 0.1, 1, 10, 100]
        tick_labels = [0.001, 0.01, 0.1, 1, 10, 100]

        # Set the tick positions and labels
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Total computation time (s)')
    else:
        # Remove existing tick labels
        plt.tick_params(axis='x', which='minor', labelbottom=False)
        # Define desired tick positions and labels
        tick_positions = [2048, 4096, 6144, 8192, 12288, 16384]
        tick_labels = [2048, 4096, 6144, 8192, 12288, 16384]
        # Set the tick positions and labels
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Sample size n')
    plt.ylabel('Power')

    if plot_time:
        plt.xlim([0.0002,370])
    else:
        plt.xlim([1536,40960])

    plt.legend(handletextpad=0.0, loc='upper left')

def format_int_list(mylist):
    formatted_list = ""
    for i in range(len(mylist)-1):
        formatted_list += str(mylist[i]) + '_'
    formatted_list += str(mylist[len(mylist)-1])
    return formatted_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMD tests figures')
    
    #General arguments
    parser.add_argument('--name', default='gaussians', help='experiment name')
    parser.add_argument('--n', type=int, default=262144, help='number of samples')
    parser.add_argument('--d', type=int, default=49, help='dimension')
    parser.add_argument('--B', type=int, default=39, help='number of permutations/Rademacher variables used')
    parser.add_argument('--alpha', type=float, default=0.05, help='level of the test')
    parser.add_argument('--n_tests', type=int, default=1, help='number of tests')
    parser.add_argument('--total_n_tests', type=int, default=200, help='number of tests')
    parser.add_argument('--error_bars_percentile', action='store_true', help='use percentiles 10 and 90 for statistics error bars')
    parser.add_argument('--no_nominal_level', action='store_true', help='if passed do not plot nominal test level (alpha)')
    parser.add_argument('--log_time_scale', action='store_true', help='if passed use log scale for time (x axis)')
    parser.add_argument('--small_times', action='store_true', help='if passed restrict time axis to the first 1200 seconds')
    parser.add_argument('--long_times', type=float, default=0.0, help='if passed restrict time axis to more than this value')
    parser.add_argument('--wilson_intervals', action='store_true', help='if passed, use Wilson confidence intervals in plots')
    parser.add_argument('--wilson_size', type=float, default=0.95, help='size of Wilson intervals')
    parser.add_argument('--no_violations', action='store_true', help='if passed, remove points for asymp. tests for which the nominal level is not respected')
    parser.add_argument('--show_exact', action='store_true', help='if passed, show only permutations/WB test for size plots')
    parser.add_argument('--test_type', type=str, default='two_sample', help='experiment name')
    parser.add_argument('--no_title', action='store_true', help='if passed, do not show title on plot')
    parser.add_argument('--n_lists', action='store_true', help='use s_lists for n_samples plots')

    #Argument for gaussians
    parser.add_argument('--mean_diff', type=float, default=0.024, help='covariance eigenvalue (for blobs)')
    
    #Arguments for blobs
    parser.add_argument('--grid_size', type=int, default=3, help='dimension of the grid of the distribution (for blobs)')
    parser.add_argument('--epsilon', type=float, default=2, help='covariance eigenvalue (for blobs)')
    
    #Arguments for MNIST and EMNIST
    parser.add_argument('--p_even', type=float, default=0.49, help='joint probability of all even digits (for MNIST and EMNIST)')
    
    #Arguments for Higgs
    parser.add_argument('--mixing', action='store_true', help='if passed use test mixing between classes')
    parser.add_argument('--null', action='store_true', help='if passed use null hypothesis, else use alternative')
    parser.add_argument('--n_components', type=int, default=4, help='number of dimensions to use')
    parser.add_argument('--p_poisoning', type=float, default=0.9, help='poisoning probability of class 1 with class 0')
    
    #Argument for gaussians_cov
    parser.add_argument('--cross_covariance', type=float, default=0.1, help='cross covariance')
    
    #Arguments to avoid plotting specific test groups
    parser.add_argument('--no_complete', action='store_true', help='if passed do not compute complete tests')
    parser.add_argument('--no_cheap_perm', action='store_true', help='if passed do not compute cheap_perm tests')
    parser.add_argument('--no_cross_mmd', action='store_true', help='if passed do not compute cross_mmd tests')
    parser.add_argument('--no_rff', action='store_true', help='if passed do not compute rff tests')
    parser.add_argument('--no_cheap_rff', action='store_true', help='if passed do not compute cheap_rff tests')
    parser.add_argument('--no_wilcoxon', action='store_true', help='if passed do not compute Wilcoxon tests')
    parser.add_argument('--no_ind_complete_WB', action='store_true', help='if passed do not compute complete tests')
    parser.add_argument('--no_ind_cheap_perm_WB', action='store_true', help='if passed do not compute cheap permutation tests')
    parser.add_argument('--no_ind_cross', action='store_true', help='if passed do not compute cross HSIC tests')
    
    parser.add_argument('--plot_n_permutations', action='store_true', help='if passed show n_permutations in x axis')
    parser.add_argument('--plot_n_samples', action='store_true', help='if passed show n_samples lines')
    parser.add_argument('--plot_n_samples_vs_power', action='store_true', help='if passed show n_samples in x axis')
    parser.add_argument('--plot_wilcoxon', action='store_true', help='if passed show Wilcoxon plot')
    parser.add_argument('--no_point_label_cheap', action='store_true', help='if passed show no point labels for cheap permutations lines')
    
    args = parser.parse_args()
    
    if args.plot_wilcoxon:
        args.interactive = False

        util_tests_cheap.get_attributes_two_sample_tests(args)
        
        test_groups = ['wilcoxon']
        
        no_compute = dict()
        no_compute['wilcoxon'] = args.no_wilcoxon
    
    elif args.test_type == 'two_sample':
        #Reset default values depending on args.name
        if args.name == 'blobs':
            args.d = 2

        if args.name == 'Higgs':
            args.d = args.n_components
            if args.p_poisoning > 0:
                args.mixing = True

        if args.name == 'sine':
            args.d = 10

        args.interactive = False

        util_tests_cheap.get_attributes_two_sample_tests(args)

        #Build list of test groups
        test_groups = ['complete', 'cheap_perm', 'cross_mmd', 'rff', 'cheap_rff']

        #Store no-compute choices for each test group
        no_compute = dict()
        no_compute['complete'] = args.no_complete
        no_compute['cheap_perm'] = args.no_cheap_perm
        no_compute['cross_mmd'] = args.no_cross_mmd
        no_compute['rff'] = args.no_rff
        no_compute['cheap_rff'] = args.no_cheap_rff
        
    elif args.test_type == 'independence':
        args.interactive = False

        util_tests_cheap.get_attributes_independence_tests(args)

        #Build list of test groups
        test_groups = ['ind_complete_WB','ind_cheap_perm_WB','ind_cross']

        #Store no-compute choices for each test group
        no_compute = dict()
        no_compute['ind_complete_WB'] = args.no_ind_complete_WB
        no_compute['ind_cheap_perm_WB'] = args.no_ind_cheap_perm_WB
        no_compute['ind_cross'] = args.no_ind_cross
    
    for group in test_groups:
        print(f'{group}: no_compute={no_compute[group]}')
        
    used_test_groups = []
    for group in test_groups:
        if not no_compute[group]:
            used_test_groups.append(group)
    if args.log_time_scale:
        used_test_groups.append('log_time_scale')
    if args.small_times:
        used_test_groups.append('small_times')
    if args.wilson_intervals:
        used_test_groups.append('wilson')
    if args.long_times != 0:
        if args.long_times.is_integer():
            str_long_times = str(int(args.long_times))
        else:
            str_long_times = str(args.long_times)
        used_test_groups.append('long_times'+'_'+str_long_times)
    formatted_used_test_groups = util_classes.format_int_list(used_test_groups)
    
    if args.plot_n_permutations:
        joint_group_results_dict = dict()
        for n_perm in args.n_permutations_list:
            args.B = n_perm
            print(f'Number of permutations: {args.B}')
            
            joint_resdir = util_classes.get_joint_group_directories(args, test_groups)

            joint_filename = util_classes_cheap.get_joint_filename(args, test_groups)

            joint_fname = util_classes.get_fname_joint(args,test_groups,joint_resdir,joint_filename)

            for group in test_groups:
                if not no_compute[group]:
                    print(f'joint_fname[group]: {joint_fname[group]}')
                    joint_group_results_dict[str(n_perm)+'_'+group] = pickle.load(open(joint_fname[group], 'rb'))

        if not os.path.exists('figures_cheap_B'+'_'+args.name):
            os.makedirs('figures_cheap_B'+'_'+args.name)
        
        plt.figure(figsize=(4.0,4.0))

        if args.test_type == 'two_sample':
            fig_B_two_sample(args, test_groups, joint_group_results_dict)
        elif args.test_type == 'independence':
            fig_B_independence(args, test_groups, joint_group_results_dict)
        
        if args.name == 'gaussians':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mean_diff)+'_'+formatted_used_test_groups+'.pdf'

        elif args.name == 'blobs':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.grid_size)+'_'+str(args.epsilon)+'_'+formatted_used_test_groups+'.pdf' 

        elif args.name == 'MNIST' or args.name == 'EMNIST':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.p_even)+'_'+formatted_used_test_groups+'.pdf' 

        elif args.name == 'Higgs':
            if args.mixing:
                fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.p_poisoning)+'_'+formatted_used_test_groups+'.pdf'
            else:
                fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.null)+'_'+formatted_used_test_groups+'.pdf'
                
        if args.name == 'gaussians_cov':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.cross_covariance)+'_'+formatted_used_test_groups+'.pdf'

        #Here new
        pplot()
        plt.tight_layout()  
        #End of new

        print('Figure file:'+'figures_cheap_B'+'_'+args.name+'/' +fig_file)

        plt.savefig(f'figures_cheap_B'+'_'+args.name+'/'+fig_file, bbox_inches='tight', pad_inches=0)
    
    elif (args.plot_n_samples or args.plot_n_samples_vs_power) and not args.plot_wilcoxon:
        joint_group_results_dict = dict()
        print(f'args.n_samples_list: {args.n_samples_list}')
        for n_samples in args.n_samples_list:
            args.n = n_samples
            args.complete_list = [args.n]
            args.cross_mmd_list = [args.n]
            args.cross_hsic_list = [args.n]
            print(f'Number of samples: {args.n}')
            
            joint_resdir = util_classes.get_joint_group_directories(args, test_groups)

            joint_filename = util_classes_cheap.get_joint_filename(args, test_groups)

            joint_fname = util_classes.get_fname_joint(args,test_groups,joint_resdir,joint_filename)

            for group in test_groups:
                if not no_compute[group]:
                    print(f'joint_fname[group]: {joint_fname[group]}')
                    joint_group_results_dict[str(n_samples)+'_'+group] = pickle.load(open(joint_fname[group], 'rb'))
        
        if args.plot_n_samples:
            if not os.path.exists('figures_cheap_n'+'_'+args.name):
                os.makedirs('figures_cheap_n'+'_'+args.name)
        elif args.plot_n_samples_vs_power:
            if not os.path.exists('figures_cheap_n_vs_power'+'_'+args.name):
                os.makedirs('figures_cheap_n_vs_power'+'_'+args.name)

        plt.figure(figsize=(4.0,4.0))
        
        if args.test_type == 'two_sample':
            if args.plot_n_samples:
                fig_n_two_sample(args, test_groups, joint_group_results_dict)
            else:
                fig_n_two_sample(args, test_groups, joint_group_results_dict, plot_time=False)
        elif args.test_type == 'independence':
            if args.plot_n_samples:
                fig_n_independence(args, test_groups, joint_group_results_dict)
            elif args.plot_n_samples_vs_power:
                fig_n_independence(args, test_groups, joint_group_results_dict, plot_time=False)
        
        if args.name == 'gaussians':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mean_diff)+'_'+formatted_used_test_groups+'.pdf'

        elif args.name == 'blobs':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.grid_size)+'_'+str(args.epsilon)+'_'+formatted_used_test_groups+'.pdf' 

        elif args.name == 'MNIST' or args.name == 'EMNIST':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.p_even)+'_'+formatted_used_test_groups+'.pdf' 

        elif args.name == 'Higgs':
            if args.mixing:
                fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.p_poisoning)+'_'+formatted_used_test_groups+'.pdf'
            else:
                fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.null)+'_'+formatted_used_test_groups+'.pdf'
                
        if args.name == 'gaussians_cov':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.cross_covariance)+'_'+formatted_used_test_groups+'.pdf'

        #Here new
        pplot()
        plt.tight_layout()  
        #End of new

        if args.plot_n_samples:
            print('Figure file:'+'figures_cheap_n'+'_'+args.name+'/' +fig_file)
            plt.savefig(f'figures_cheap_n'+'_'+args.name+'/'+fig_file, bbox_inches='tight', pad_inches=0)
        elif args.plot_n_samples_vs_power:
            print('Figure file:'+'figures_cheap_n_vs_power'+'_'+args.name+'/' +fig_file)
            plt.savefig(f'figures_cheap_n_vs_power'+'_'+args.name+'/'+fig_file, bbox_inches='tight', pad_inches=0)
        
    elif args.plot_wilcoxon:
        if args.plot_n_samples or args.plot_n_samples_vs_power:
            test_groups = ['wilcoxon']
            joint_group_results_dict = dict()
            for n_samples in args.n_samples_list_wilcoxon:
                args.n = n_samples
                args.wilcoxon_list_n[-1] = n_samples
                print(f'Number of samples: {args.n}')

                joint_resdir = util_classes.get_joint_group_directories(args, test_groups)

                joint_filename = util_classes_cheap.get_joint_filename(args, test_groups)

                joint_fname = util_classes.get_fname_joint(args,test_groups,joint_resdir,joint_filename)

                for group in test_groups:
                    if not no_compute[group]:
                        print(f'joint_fname[group]: {joint_fname[group]}')
                        joint_group_results_dict[str(n_samples)+'_'+group] = pickle.load(open(joint_fname[group], 'rb'))
        else:
            test_groups = ['wilcoxon']
        
            joint_resdir = util_classes.get_joint_group_directories(args, test_groups)

            joint_filename = util_classes_cheap.get_joint_filename(args, test_groups)

            joint_fname = util_classes.get_fname_joint(args,test_groups,joint_resdir,joint_filename)

            joint_group_results_dict = dict()
        
            for group in test_groups:
                if not no_compute[group]:
                    print(f'joint_fname[group]: {joint_fname[group]}')
                    joint_group_results_dict[group] = pickle.load(open(joint_fname[group], 'rb'))
        
        if args.plot_n_samples:
            if not os.path.exists('figures_cheap_w_n'+'_'+args.name):
                os.makedirs('figures_cheap_w_n'+'_'+args.name)
        elif args.plot_n_samples_vs_power:
            if not os.path.exists('figures_cheap_w_n_samples_vs_power'+'_'+args.name):
                os.makedirs('figures_cheap_w_n_samples_vs_power'+'_'+args.name)
        else:
            if not os.path.exists('figures_cheap_w'+'_'+args.name):
                os.makedirs('figures_cheap_w'+'_'+args.name)
            
        plt.figure(figsize=(4.0,4.0))
        if args.plot_n_samples:
            fig_n_Wilcoxon(args, test_groups, joint_group_results_dict)
        elif args.plot_n_samples_vs_power:
            fig_n_Wilcoxon(args, test_groups, joint_group_results_dict, plot_time=False)
        else:
            fig_Wilcoxon_two_sample(args, test_groups, joint_group_results_dict)
        
        if args.name == 'gaussians':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mean_diff)+'_'+formatted_used_test_groups+'.pdf'
            
        #Here new
        pplot()
        plt.tight_layout()  
        #End of new

        if args.plot_n_samples:
            print('Figure file:'+'figures_cheap_w_n'+'_'+args.name+'/' +fig_file)
            plt.savefig(f'figures_cheap_w_n'+'_'+args.name+'/'+fig_file, bbox_inches='tight', pad_inches=0)
        elif args.plot_n_samples_vs_power:
            print('Figure file:'+'figures_cheap_w_n_samples_vs_power'+'_'+args.name+'/' +fig_file)
            plt.savefig(f'figures_cheap_w_n_samples_vs_power'+'_'+args.name+'/'+fig_file, bbox_inches='tight', pad_inches=0)
        else:
            print('Figure file:'+'figures_cheap_w'+'_'+args.name+'/' +fig_file)
            plt.savefig(f'figures_cheap_w'+'_'+args.name+'/'+fig_file, bbox_inches='tight', pad_inches=0)
        
    else:
        joint_resdir = util_classes.get_joint_group_directories(args, test_groups)

        joint_filename = util_classes_cheap.get_joint_filename(args, test_groups)

        joint_fname = util_classes.get_fname_joint(args,test_groups,joint_resdir,joint_filename)

        joint_group_results_dict = dict()

        for group in test_groups:
            if not no_compute[group]:
                print(f'joint_fname[group]: {joint_fname[group]}')
                joint_group_results_dict[group] = pickle.load(open(joint_fname[group], 'rb'))

        if not os.path.exists('figures_cheap'+'_'+args.name):
            os.makedirs('figures_cheap'+'_'+args.name)


        if args.test_type == 'two_sample':
            if not args.no_cheap_rff:
                plt.figure(figsize=(5.5,3.5))
            else:
                plt.figure(figsize=(4.0,4.0))
            fig_rejection_two_sample(args, test_groups, joint_group_results_dict)
        elif args.test_type == 'independence':
            plt.figure(figsize=(4.0,4.0))
            fig_rejection_independence(args, test_groups, joint_group_results_dict)

        if args.name == 'gaussians':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mean_diff)+'_'+formatted_used_test_groups+'.pdf'

        elif args.name == 'blobs':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.grid_size)+'_'+str(args.epsilon)+'_'+formatted_used_test_groups+'.pdf' 

        elif args.name == 'MNIST' or args.name == 'EMNIST':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.p_even)+'_'+formatted_used_test_groups+'.pdf' 

        elif args.name == 'Higgs':
            if args.mixing:
                fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.p_poisoning)+'_'+formatted_used_test_groups+'.pdf'
            else:
                fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.mixing)+'_'+str(args.null)+'_'+formatted_used_test_groups+'.pdf'
                
        elif args.name == 'gaussians_cov':
            fig_file = 'rejection_probability_'+args.name+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+str(args.cross_covariance)+'_'+formatted_used_test_groups+'.pdf'

        #Here new
        pplot()
        plt.tight_layout()  
        #End of new

        print('Figure file:'+'figures_cheap'+'_'+args.name+'/' +fig_file)

        plt.savefig(f'figures_cheap'+'_'+args.name+'/'+fig_file, bbox_inches='tight', pad_inches=0)
