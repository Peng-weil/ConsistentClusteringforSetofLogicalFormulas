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


def construct_x_domain(N, r):
    C = [0]
    for i in range(1, r):
        C.append(i)
    C.append(N)
    return C


parser = argparse.ArgumentParser(description='Consistent Clustering for Set of Logical Formulas')
parser.add_argument('--path', type=str, default='./mups/results-mups-real/CHEM-A/res.txt', help='mups path')
parser.add_argument('--log', type=str, default='./mups/results-mups-real/CHEM-A/log-alg2.txt', help='log file')
parser.add_argument('--r', type=int, default=2, help='max Min(K,phi)')

if __name__ == '__main__':
    args = parser.parse_args()
    r = args.r
    # ALG2
    mis_path = args.path
    log_path = args.log

    # f = open(log_path, 'w')
    # sys.stdout = f
    # context.log_output = f

    mis_set, formula_dict = load_mups_formula(mis_path)
    m = len(mis_set)
    n = len(formula_dict)

    N = n + 1
    x_domain_list = construct_x_domain(N, r)

    model = CpoModel('Consistent Clustering')
    mdl_x_dict = {}
    for formula_value in formula_dict.values():
        mdl_x_dict[formula_value] = model.integer_var(domain=tuple(x_domain_list), name=formula_value)

    mdl_q_dict = {}
    for i in range(m):
        var_q = 'q_' + str(i)
        mdl_q_dict[var_q] = model.integer_var(0, 1, var_q)

    mdl_y_dict = {}
    for i in range(m):
        for j in range(n):
            var_y = 'y_' + str(i) + '_' + str(j)
            mdl_y_dict[var_y] = model.integer_var(0, 1, var_y)

    x_sum = None
    for x in mdl_x_dict.values():
        x_sum = x if x_sum is None else x_sum + x

    for i in range(len(mis_set)):
        yit_sum = []
        mis_single = mis_set[i]
        bi = len(mis_single)

        mis_single_sum = []
        for pb_formula in mis_single:
            x_oi = mdl_x_dict[formula_dict[pb_formula]]
            mis_single_sum.append(x_oi)

        for t in range(len(list(mis_single))):
            xt = mdl_x_dict[formula_dict[list(mis_single)[t]]]
            tfy = xt.name.split('_')[1]
            yit = mdl_y_dict['y_' + str(i) + '_' + str(tfy)]
            yit_sum.append(yit)
            model.add(model.sum(mis_single_sum) > yit * xt * bi)

        qi = mdl_q_dict['q_' + str(i)]
        model.add(model.sum(yit_sum) >= qi)
        model.add(model.sum(mis_single_sum) >= bi * r * (1 - qi))

    model.add(model.minimize(x_sum))
    result = model.solve()

    print('\n')
    for f_k in formula_dict.keys():
        print('{}: {}'.format(formula_dict[f_k], f_k))
    print('\n')

    print('problem_formula: {}\n'
          'mis_size: {}\n'
          'r: {}\n'.format(n, m, r))

    cluster = {}
    if result:
        for x in mdl_x_dict:
            if result[x] not in cluster:
                cluster[result[x]] = [x]
            else:
                cluster[result[x]].append(x)

    for c in cluster.keys():
        print('{}:{}\n'.format(c, cluster[c]))

