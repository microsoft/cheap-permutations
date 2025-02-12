import numpy as np
import os
import glob
import argparse
import pickle
import util_classes
import util_classes_cheap
import util_tests_cheap


def get_results(args):
    resdir = util_classes_cheap.get_group_directories(args, test_groups)
            
    groupname, testname = util_classes_cheap.get_test_file_names(args, test_groups, resdir, save=False)
    
    joint_resdir = util_classes.get_joint_group_directories(args, test_groups)
    
    joint_filename = util_classes_cheap.get_joint_filename(args, test_groups)
    
    joint_fname = util_classes.get_fname_joint(args, test_groups, joint_resdir, joint_filename)
    
    fnames = dict()
    file_exists = dict()
    joint_group_results_dict = dict()
    
    for group in test_groups:
        print(f'group: {group}, no_compute: {no_compute[group]}')
        if not no_compute[group]:
            name = os.path.join(resdir[group],groupname[group])
            print(f'File to retrieve for {group}: {name}')
            fnames[group] = glob.glob(name)
            assert len(fnames[group]) > 0, 'no files! ({})'.format(name)
            if len(fnames[group]) and not args.interactive:
                file_exists[group] = True
                print(f'{len(fnames[group])} files exist')
            else:
                file_exists[group] = False

            total_n_tests_found = len(fnames[group])
            #print(f'{args.n_tests}x{len(fnames[group])} = {total_n_tests_found} tests found. {args.total_n_tests} needed.')
            print(f'{total_n_tests_found} tests found. {args.total_n_tests} needed.')

            joint_group_results_dict[group] = util_classes.JointGroupResults(joint_fname[group], args.total_n_tests, args.B, 1, file_exists[group])

            joint_group_results_dict[group].set_compute(no_compute[group])

            #print(f'fnames[group]: {fnames[group]}')
            for i, fname in enumerate(fnames[group]):
                if i >= args.total_n_tests:
                    print(f'{i}/{args.total_n_tests} processed, all remaining tests are disregarded.')
                    break
                if os.path.getsize(fname) > 0:
                    res = pickle.load(open(fname, 'rb'))
                else:
                    print(f'File empty: {fname}')

                #print(f'joint_group_results_dict[group].group_names: {joint_group_results_dict[group].group_names}')
                if joint_group_results_dict[group].group_names is None:
                    group_names = res['group_names']
                    full_group_names = res['full_group_names']
                    group_labels = res['group_labels']
                    joint_group_results_dict[group].set_group_names(group_names, full_group_names, group_labels)
                    #print('set_group_names done')
                       
                if joint_group_results_dict[group].bw is None:
                    joint_group_results_dict[group].set_bandwidth(res['bw'])

                joint_group_results_dict[group].update_attributes(res)

            joint_group_results_dict[group].compute_info()

            joint_group_results_dict[group].print_info()

            joint_group_results_dict[group].save()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMD tests results')
    
    #General arguments
    parser.add_argument('--name', default='gaussians', help='experiment name')
    parser.add_argument('--n', type=int, default=262144, help='number of samples')
    parser.add_argument('--d', type=int, default=49, help='dimension')
    parser.add_argument('--B', type=int, default=39, help='number of permutations/Rademacher variables used')
    parser.add_argument('--alpha', type=float, default=0.05, help='level of the test')
    parser.add_argument('--n_tests', type=int, default=1, help='number of tests')
    parser.add_argument('--total_n_tests', type=int, default=200, help='number of tests')
    parser.add_argument('--test_type', type=str, default='two_sample', help='experiment name')
    parser.add_argument('--n_lists', action='store_true', help='use s_lists for n_samples plots')

    #Argument for gaussians
    parser.add_argument('--mean_diff', type=float, default=0.024, help='mean difference (for gaussians)')
    
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
    
    #Arguments to avoid postprocessing specific test groups
    parser.add_argument('--no_complete', action='store_true', help='if passed do not compute complete tests')
    parser.add_argument('--no_cheap_perm', action='store_true', help='if passed do not compute cheap_perm tests')
    parser.add_argument('--no_cross_mmd', action='store_true', help='if passed do not compute cross_mmd tests')
    parser.add_argument('--no_rff', action='store_true', help='if passed do not compute rff tests')
    parser.add_argument('--no_cheap_rff', action='store_true', help='if passed do not compute cheap_rff tests')
    parser.add_argument('--no_wilcoxon', action='store_true', help='if passed do not compute Wilson tests')
    
    parser.add_argument('--no_ind_complete_WB', action='store_true', help='if passed do not compute complete tests')
    parser.add_argument('--no_ind_cheap_perm_WB', action='store_true', help='if passed do not compute cheap_rff tests')
    parser.add_argument('--no_ind_cross', action='store_true', help='if passed do not compute cross HSIC tests')
    args = parser.parse_args()
    
    if args.test_type == 'two_sample':
        #Reset default values depending on args.name
#         if args.name == 'gaussians':
#             args.d = 10

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

        #Store no-compute choices for each test group
        no_compute = dict()
        no_compute['complete'] = args.no_complete
        no_compute['cheap_perm'] = args.no_cheap_perm
        no_compute['cross_mmd'] = args.no_cross_mmd
        no_compute['rff'] = args.no_rff
        no_compute['cheap_rff'] = args.no_cheap_rff
        no_compute['wilcoxon'] = args.no_wilcoxon

        #Build list of test groups
        test_groups = ['complete', 'cheap_perm', 'cross_mmd', 'rff', 'cheap_rff', 'wilcoxon']
        args.estimator_list = test_groups
        
    elif args.test_type == 'independence':
        # if args.name == 'gaussians_cov':
        #    args.d = 10
            
        args.interactive = False
        
        util_tests_cheap.get_attributes_independence_tests(args)
        
        #Store no-compute choices for each test group
        no_compute = dict()
        no_compute['ind_complete_WB'] = args.no_ind_complete_WB
        no_compute['ind_cheap_perm_WB'] = args.no_ind_cheap_perm_WB
        no_compute['ind_cross'] = args.no_ind_cross
        
        #Build list of test groups
        test_groups = ['ind_complete_WB', 'ind_cheap_perm_WB', 'ind_cross']
        args.estimator_list = test_groups
        
    get_results(args)
    
