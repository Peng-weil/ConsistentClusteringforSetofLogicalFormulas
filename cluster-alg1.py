import copy
import sys
import argparse

from docplex.cp.model import CpoModel
from docplex.cp.config import context


def load_mups_formula(mups_path):
    mups_set = []
    mups_single = []
    formula_dict = {}

    mups_file = open(mups_path)
    line = mups_file.readline()
    mups_start = False
    var_x = 0
    while line:
        if (not line.strip().startswith('Found explanation <')) and (not line.strip().startswith('Explanation <')):
            if len(line.strip()) != 0 and line.strip().startswith('['):
                if mups_start:
                    mups_str = line[line.find(']') + 1:len(line)].strip()
                    if not mups_str in mups_single:
                        mups_single.append(mups_str)
                    if mups_str not in formula_dict.keys():
                        formula_dict[mups_str] = 'x_' + str(var_x)
                        var_x = var_x + 1
            else:
                mups_start = False
                if not len(mups_single) == 0:
                    if not set(mups_single) in mups_set:
                        mups_set.append(copy.deepcopy(set(mups_single)))
                    mups_single.clear()
        else:
            mups_start = True
        line = mups_file.readline()
    mups_file.close()
    return mups_set, formula_dict


def construct_x_domain(N, n):
    C = [0]
    for i in range(n - 1):
        extension_x = pow(N, i)
        C.append(extension_x)
    return C


parser = argparse.ArgumentParser(description='Consistent Clustering for Set of Logical Formulas')
parser.add_argument('--path', type=str, default='./mups/results-mups-real/buggyPolicy/res.txt', help='mups path')
parser.add_argument('--log', type=str, default='./mups/results-mups-real/buggyPolicy/log-alg1.txt', help='log file')

if __name__ == '__main__':
    result = False
    iteration = 1
    args = parser.parse_args()
    # ALG1
    mis_path = args.path
    log_path = args.log

    f = open(log_path, 'w')
    sys.stdout = f
    context.log_output = f

    # line1
    mis_set, formula_dict = load_mups_formula(mis_path)
    # line2
    m = len(mis_set)
    n = len(formula_dict)
    # line3
    N = n + 1

    x_domain_list = construct_x_domain(N, n)

    while not result:
        model = CpoModel('Consistent Clustering')
        iteration = iteration + 1
        x_domain = x_domain_list[:iteration]

        mdl_x_dict = {}

        # line5, line6-line8
        for formula_value in formula_dict.values():
            mdl_x_dict[formula_value] = model.integer_var(domain=tuple(x_domain), name=formula_value)

        # line9, line10-line12
        mdl_y_dict = {}
        for i in range(m):
            for j in range(n):
                var_y = 'y_' + str(i) + '_' + str(j)
                mdl_y_dict[var_y] = model.integer_var(0, 1, var_y)

        # line13
        x_sum = None
        for x in mdl_x_dict.values():
            x_sum = x if x_sum is None else x_sum + x

        # line14
        for i in range(len(mis_set)):
            yit_sum = []
            # line14
            mis_single = mis_set[i]
            bi = len(mis_single)

            mis_single_sum = []
            for pb_formula in mis_single:
                x_oi = mdl_x_dict[formula_dict[pb_formula]]
                mis_single_sum.append(x_oi)

            # line15-line18
            for t in range(len(list(mis_single))):
                xt = mdl_x_dict[formula_dict[list(mis_single)[t]]]
                tfy = xt.name.split('_')[1]
                yit = mdl_y_dict['y_' + str(i) + '_' + str(tfy)]
                yit_sum.append(yit)
                model.add(model.sum(mis_single_sum) > yit * xt * bi)

            # line19
            model.add(model.sum(yit_sum) > 0)

        # line21
        model.add(model.minimize(x_sum))
        result = model.solve()

        if result:
            print('\n')
            for f_k in formula_dict.keys():
                print('{}: {}'.format(formula_dict[f_k], f_k))
            print('\n')

            cluster = {}
            for x in mdl_x_dict:
                if result[x] not in cluster:
                    cluster[result[x]] = [x]
                else:
                    cluster[result[x]].append(x)

            max_c = 0
            for c in cluster.keys():
                max_c = c if c >= max_c else max_c

            print('problem_formula: {}\n'
                  'mis_size: {}\n'
                  'iteration: {}\n'
                  'r: {}\n'.format(n, m, iteration, max_c + 1))

            for c in cluster.keys():
                print('{}:{}\n'.format(c, cluster[c]))
