import numpy as np

def get_attributes_two_sample_tests(args):
    #Build list of test groups
    test_groups = ['complete', 'cheap_perm', 'cross_mmd', 'rff', 'cheap_rff', 'wilcoxon']
    
    #Define dicts to store estimator, estimator names (used when printing results), and estimator labels (used in plots)
    args.estimators = dict()
    args.estimator_names = dict()
    args.estimator_labels = dict()
    for group in test_groups:
        args.estimators[group] = []
        args.estimator_names[group] = []
        args.estimator_labels[group] = []
        
    #Build list of estimators for complete tests
    args.complete_list = [args.n] 
    for n in args.complete_list:
        args.estimators['complete'].append(str(n))
        args.estimator_names['complete'].append('complete '+str(n))
        args.estimator_labels['complete'].append('n='+str(n))
        
    #Build list of estimators for cheap permutation tests
    args.cheap_perm_list = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    args.cheap_perm_list_n = [4,8,16,32,64,128,256,512] #[4,8,32,128,512] 
    for s in args.cheap_perm_list:
        args.estimators['cheap_perm'].append(str(s))
        args.estimator_names['cheap_perm'].append('cheap perm. '+str(s))
        args.estimator_labels['cheap_perm'].append('s='+str(s))
        
    #Build list of estimators for cross MMD tests   
    args.cross_mmd_list = [args.n]
    for n in args.cross_mmd_list:
        args.estimators['cross_mmd'].append(str(n))
        args.estimator_names['cross_mmd'].append('cross MMD '+str(n))
        args.estimator_labels['cross_mmd'].append('n='+str(n))
        
    #Build list of estimators for random Fourier features tests   
    args.rff_list = [8,32,128,512,2048]
    for r in args.rff_list:
        args.estimators['rff'].append(str(r))
        args.estimator_names['rff'].append('RFF '+str(r))
        args.estimator_labels['rff'].append('r='+str(r))
        
    #Build list of estimators for random Fourier features tests   
    args.cheap_rff_list_n_features = [8,32,128,512,2048]
    args.cheap_rff_list_s = [8,16,32,128,512,2048,8192]
    for r in args.cheap_rff_list_n_features:
        for s in args.cheap_rff_list_s:
            args.estimators['cheap_rff'].append(str(r)+'_'+str(s))
            args.estimator_names['cheap_rff'].append('Cheap RFF, r='+str(r)+', s='+str(s))
            args.estimator_labels['cheap_rff'].append('r='+str(r)+', s='+str(s))
            
    #Build list of estimators for Wilcoxon tests
    args.wilcoxon_list = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    args.wilcoxon_list_n = [4,8,16,32,64,128,256,512,args.n]
    wilcoxon_list = args.wilcoxon_list_n if args.n_lists else args.wilcoxon_list
    for s in wilcoxon_list:
        args.estimators['wilcoxon'].append(str(s))
        args.estimator_names['wilcoxon'].append('Wilcoxon '+str(s))
        args.estimator_labels['wilcoxon'].append('s='+str(s))
            
    #List of number of permutations
    args.n_permutations_list = [19,39,79,159,319,639,1279,2559]
    
    #List of number of samples
    args.n_samples_list = [1024,2048,3072,4096,6144,8192,12288,16384]

    #List of number of samples (Wilcoxon)
    args.n_samples_list_wilcoxon = [1024,2048,3072,4096,6144,8192]
    
def get_attributes_independence_tests(args):
    #Build list of test groups
    test_groups = ['ind_complete_WB','ind_cheap_perm_WB','ind_cross']
    
    #Define dicts to store estimator, estimator names (used when printing results), and estimator labels (used in plots)
    args.estimators = dict()
    args.estimator_names = dict()
    args.estimator_labels = dict()
    for group in test_groups:
        args.estimators[group] = []
        args.estimator_names[group] = []
        args.estimator_labels[group] = []
        
    #Build list of estimators for complete tests
    args.complete_list = [args.n]
    for n in args.complete_list:
        args.estimators['ind_complete_WB'].append(str(n))
        args.estimator_names['ind_complete_WB'].append('complete WB '+str(n))
        args.estimator_labels['ind_complete_WB'].append('n='+str(n))
    
    #Build list of estimators for cheap permutation tests   
    args.cheap_perm_list = [4,8,16,32,64,128,256,512,1024,2048]
    args.cheap_perm_list_n = [4,8,16,64,128,256,512]
    for s in args.cheap_perm_list:
        args.estimators['ind_cheap_perm_WB'].append(str(s))
        args.estimator_names['ind_cheap_perm_WB'].append('cheap perm. '+str(s))
        args.estimator_labels['ind_cheap_perm_WB'].append('s='+str(s))
        
    #Build list of estimators for cross MMD tests   
    args.cross_hsic_list = [args.n]
    for n in args.cross_hsic_list:
        args.estimators['ind_cross'].append(str(n))
        args.estimator_names['ind_cross'].append('cross HSIC '+str(n))
        args.estimator_labels['ind_cross'].append('n='+str(n))
        
    #List of number of permutations
    args.n_permutations_list = [19,39,79,159,319,639,1279,2559]
    
    #List of number of samples
    args.n_samples_list = [512,1024,1536,2048,3072,4096]
