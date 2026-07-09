
import os

def format_int_list(mylist):
    formatted_list = ""
    for i in range(len(mylist)-1):
        formatted_list += str(mylist[i]) + '_'
    formatted_list += str(mylist[len(mylist)-1])
    return formatted_list

def get_group_directories(args, test_groups, aggregated=False):
    resdir = dict()
    
    if args.name == 'gaussians':
        name_arguments = str(args.mean_diff)
    elif args.name == 'blobs':
        name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
    elif args.name == 'MNIST' or args.name == 'EMNIST': 
        name_arguments = str(args.p_even)
    elif args.name == 'Higgs': 
        if args.mixing:
            name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
        else:
            name_arguments = str(args.mixing)+'_'+str(args.null)
    elif args.name == 'sine':
        name_arguments = str(args.omega)
    elif args.name == 'gaussians_cov':
        name_arguments = str(args.cross_covariance)
    else:
        name_arguments = 'None'
        
    for group in test_groups:
        #define directory to store results for estimators in group, and create it if needed 
        if aggregated:
            resdir[group] = os.path.join('res', args.name+'_'+group+'_'+name_arguments+'_aggregated')
        else:
            resdir[group] = os.path.join('res', args.name+'_'+group+'_'+name_arguments)
        if not os.path.exists(resdir[group]):
            os.makedirs(resdir[group], exist_ok=True)
            
    return resdir

def get_test_file_names(args, test_groups, resdir, n_tests=1, save=True, aggregated=False):
    #save = True to save files (in test.py), save = False to get files loaded (in postprocessing.py)
    
    if args.n_bandwidths > 1:
        # Aggregated two-sample testing
        formatted_complete_list = format_int_list(args.complete_list)
        formatted_cheap_perm_list = format_int_list(args.cheap_perm_list_n) if args.n_lists else format_int_list(args.cheap_perm_list)

        if args.name == 'gaussians':
            name_arguments = str(args.mean_diff)
        elif args.name == 'blobs':
            name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
        elif args.name == 'MNIST' or args.name == 'EMNIST': 
            name_arguments = str(args.p_even)
        elif args.name == 'Higgs': 
            if args.mixing:
                name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
            else:
                name_arguments = str(args.mixing)+'_'+str(args.null)
        elif args.name == 'sine':
            name_arguments = str(args.omega)
        else:
            name_arguments = 'None'

        group_arguments = dict()
        group_arguments['complete'] = formatted_complete_list
        group_arguments['cheap_perm'] = formatted_cheap_perm_list
    elif args.test_type == 'two_sample':   
        formatted_complete_list = format_int_list(args.complete_list)
        formatted_cheap_perm_list = format_int_list(args.cheap_perm_list_n) if args.n_lists else format_int_list(args.cheap_perm_list)
        formatted_cross_mmd_list = format_int_list(args.cross_mmd_list)
        formatted_rff_list = format_int_list(args.rff_list)
        formatted_cheap_rff_list_n_features = format_int_list(args.cheap_rff_list_n_features)
        formatted_cheap_rff_list_s = format_int_list(args.cheap_rff_list_s)
        formatted_wilcoxon_list = format_int_list(args.wilcoxon_list_n) if args.n_lists else format_int_list(args.wilcoxon_list) 

        if args.name == 'gaussians':
            name_arguments = str(args.mean_diff)
        elif args.name == 'blobs':
            name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
        elif args.name == 'MNIST' or args.name == 'EMNIST': 
            name_arguments = str(args.p_even)
        elif args.name == 'Higgs': 
            if args.mixing:
                name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
            else:
                name_arguments = str(args.mixing)+'_'+str(args.null)
        elif args.name == 'sine':
            name_arguments = str(args.omega)
        else:
            name_arguments = 'None'

        group_arguments = dict()
        group_arguments['complete'] = formatted_complete_list
        group_arguments['cheap_perm'] = formatted_cheap_perm_list
        group_arguments['cross_mmd'] = formatted_cross_mmd_list
        group_arguments['rff'] = formatted_rff_list
        group_arguments['cheap_rff'] = formatted_cheap_rff_list_n_features+'_'+formatted_cheap_rff_list_s
        group_arguments['wilcoxon'] = formatted_wilcoxon_list
        
    elif args.test_type == 'independence':
        
        formatted_complete_list = format_int_list(args.complete_list)
        formatted_cheap_perm_list = format_int_list(args.cheap_perm_list_n) if args.n_lists else format_int_list(args.cheap_perm_list)
        formatted_cross_hsic_list = format_int_list(args.cross_hsic_list)
        
        if args.name == 'gaussians_cov':
            name_arguments = str(args.cross_covariance)
        
        group_arguments = dict()
        group_arguments['ind_complete_WB'] = formatted_complete_list
        group_arguments['ind_cheap_perm_WB'] = formatted_cheap_perm_list
        group_arguments['ind_cross'] = formatted_cross_hsic_list
        
    if save:
        
        groupname = [None]*n_tests
        testname = [None]*n_tests

        for i in range(n_tests):
            groupname[i] = dict()
            testname[i] = dict()

        for i in range(n_tests):
            
            seed = str(args.seed)+'_'+str(i)
            
            for group in test_groups:
                ### Remove args.n maybe
                groupname[i][group] = 'cheap'+'_'+args.name+'_'+group+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+seed +'_'+str(args.alpha)+'_'+name_arguments

                groupname_dir = os.path.join(resdir[group], groupname[i][group])
                if args.save_objects and not os.path.exists(groupname_dir):
                    os.makedirs(groupname_dir, exist_ok=True)

                testname[i][group] = dict()
                for tname in args.estimators[group]:
                    testname[i][group][tname] = os.path.join(groupname[i][group],tname)
                groupname[i][group] += '_'+group_arguments[group]

        return groupname, testname
    
    else:
        
        groupname = dict()
        testname = dict()
            
        seed = '*'

        for group in test_groups:
            ### Remove args.n maybe
            groupname[group] = 'cheap'+'_'+args.name+'_'+group+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+seed +'_'+str(args.alpha)+'_'+name_arguments
            testname[group] = dict()
            for tname in args.estimators[group]:
                testname[group][tname] = os.path.join(groupname[group],tname)
            groupname[group] += '_'+group_arguments[group]

        return groupname, testname
    
def get_joint_filename(args, test_groups, aggregated=False):
    
    joint_filename = dict() 
    
    if args.n_bandwidths > 1:
        # Aggregated two-sample testing
        formatted_complete_list = format_int_list(args.complete_list)
        formatted_cheap_perm_list = format_int_list(args.cheap_perm_list_n) if args.n_lists else format_int_list(args.cheap_perm_list)

        if args.name == 'gaussians':
            name_arguments = str(args.mean_diff)
        elif args.name == 'blobs':
            name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
        elif args.name == 'MNIST' or args.name == 'EMNIST': 
            name_arguments = str(args.p_even)
        elif args.name == 'Higgs': 
            if args.mixing:
                name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
            else:
                name_arguments = str(args.mixing)+'_'+str(args.null)
        elif args.name == 'sine':
            name_arguments = str(args.omega)
        else:
            name_arguments = 'None'

        group_arguments = dict()
        group_arguments['complete'] = formatted_complete_list
        group_arguments['cheap_perm'] = formatted_cheap_perm_list
    elif args.test_type == 'two_sample':
    
        formatted_complete_list = format_int_list(args.complete_list)
        formatted_cheap_perm_list = format_int_list(args.cheap_perm_list_n) if args.n_lists else format_int_list(args.cheap_perm_list)
        formatted_cross_mmd_list = format_int_list(args.cross_mmd_list)
        formatted_rff_list = format_int_list(args.rff_list)
        formatted_cheap_rff_list_n_features = format_int_list(args.cheap_rff_list_n_features)
        formatted_cheap_rff_list_s = format_int_list(args.cheap_rff_list_s)
        formatted_wilcoxon_list = format_int_list(args.wilcoxon_list_n) if args.n_lists else format_int_list(args.wilcoxon_list)

        if args.name == 'gaussians':
            name_arguments = str(args.mean_diff)
        elif args.name == 'blobs':
            name_arguments = str(args.grid_size)+'_'+str(args.epsilon)
        elif args.name == 'MNIST' or args.name == 'EMNIST': 
            name_arguments = str(args.p_even)
        elif args.name == 'Higgs': 
            if args.mixing:
                name_arguments = str(args.mixing)+'_'+str(args.p_poisoning)
            else:
                name_arguments = str(args.mixing)+'_'+str(args.null)
        elif args.name == 'sine':
            name_arguments = str(args.omega)
        else:
            name_arguments = 'None'

        group_arguments = dict()
        group_arguments['complete'] = formatted_complete_list
        group_arguments['cheap_perm'] = formatted_cheap_perm_list
        group_arguments['cross_mmd'] = formatted_cross_mmd_list
        group_arguments['rff'] = formatted_rff_list
        group_arguments['cheap_rff'] = formatted_cheap_rff_list_n_features+'_'+formatted_cheap_rff_list_s
        group_arguments['wilcoxon'] = formatted_wilcoxon_list
        
    elif args.test_type == 'independence':
        
        formatted_complete_list = format_int_list(args.complete_list)
        formatted_cheap_perm_list = format_int_list(args.cheap_perm_list_n) if args.n_lists else format_int_list(args.cheap_perm_list)
        formatted_cross_hsic_list = format_int_list(args.cross_hsic_list)
        
        if args.name == 'gaussians_cov':
            name_arguments = str(args.cross_covariance)
        else:
            name_arguments = 'None'
            
        group_arguments = dict()
        group_arguments['ind_complete_WB'] = formatted_complete_list
        group_arguments['ind_cheap_perm_WB'] = formatted_cheap_perm_list
        group_arguments['ind_cross'] = formatted_cross_hsic_list
    
    for group in test_groups:
        ### Remove args.n maybe
        if aggregated:
            joint_filename[group] = 'cheap'+'_'+args.name+'_'+group+'_aggregated_'+str(args.n_bandwidths)+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.B_2) +'_'+str(args.B_3)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+name_arguments+'_'+group_arguments[group]
        else:
            joint_filename[group] = 'cheap'+'_'+args.name+'_'+group+'_'+str(args.d)+'_'+str(args.n)+'_'+str(args.B)+'_'+str(args.alpha) +'_'+str(args.total_n_tests)+'_'+name_arguments+'_'+group_arguments[group]
     
    return joint_filename
