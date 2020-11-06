from TKPExperimentPlots import *

#TODO: too many TAME data processing functions, should be reduced to 1

triangle_counts = {
    "fly PHY1": 58216,
    "human Y2H1": 15177,
    "human PHY1": 525238,
    "yeast PHY1": 381812,
    "yeast Y2H1": 9503,
    "worm PHY1": 692,
    "worm Y2H1": 536,
    "yeast PHY2": 26296,
    "human PHY2": 19190,
    "fly Y2H1": 2501
}

vertex_counts = {
    "fly PHY1": 7885,
    "human Y2H1": 9996,
    "human PHY1": 16060,
    "worm PHY1": 3003,
    "yeast Y2H1": 3427,
    "yeast PHY1": 6168,
    "worm Y2H1": 2871,
    "yeast PHY2": 3768,
    "human PHY2": 8283,
    "fly Y2H1": 7094
}


#source: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
def flatten(l):
   return [item for sublist in l for item in sublist]


def get_rank_stats(exps):
    experiment_count = len(exps)
    max_iter = max([len(x) for x in exps])

    # copy data into a matrix
    Data = -np.ones((experiment_count, max_iter))

    for i in range(experiment_count):
        for j in range(len(exps[i])):  # experiments may have different lengths
            Data[i, j] = exps[i][j]

    #    All_exp_rank_elemwise_sums = reduce(list_elementwise_sum,all_ranks_all_exps)

    # compute averages and variance
    All_exp_average = []
    All_exp_20th_percentile = []
    All_exp_80th_percentile = []
    n, d = Data.shape

    for j in range(d):
        vals = []
        for i in range(n):
            if Data[i, j] != -1:
                vals.append(Data[i, j])
        #All_exp_average.append(np.mean(vals))
        All_exp_average.append(np.percentile(vals,50))
        All_exp_20th_percentile.append(np.percentile(vals,20))
        All_exp_80th_percentile.append(np.percentile(vals,80))

    return All_exp_average , All_exp_20th_percentile, All_exp_80th_percentile


#increasing triangle order: [6, 5, 1, 9, 4, 3, 8, 0, 7, 2]

def get_results(perm=None):

    graph_names = [
        "fly PHY1",
        "human Y2H1",
        "human PHY1",
        "worm PHY1",
        "yeast Y2H1",
        "yeast PHY1",
        "worm Y2H1",
        "yeast PHY2",
        "human PHY2",
        "fly Y2H1"
    ]


    n = len(graph_names)


    LGRAAL_tri_results = np.array([
        [0.000000 , 0.000000 , 0.104250 , 0.143064 , 0.142060 , 0.070479 , 0.113806 , 0.159954 , 0.044294 , 0.772091],
        [0.000000 , 0.000000 , 0.028135 , 0.166185 , 0.107335 , 0.007116 , 0.100746 , 0.000000 , 0.000000 , 0.023191],
        [0.104250 , 0.028135 , 0.000000 , 0.065029 , 0.055561 , 0.115305 , 0.033582 , 0.189808 , 0.150651 , 0.013195],
        [0.143064 , 0.166185 , 0.065029 , 0.000000 , 0.115607 , 0.044798 , 1.000000 , 0.206647 , 0.125723 , 0.044798],
        [0.142060 , 0.107335 , 0.055561 , 0.115607 , 0.000000 , 0.047459 , 0.123134 , 0.056824 , 0.044723 , 0.000000],
        [0.070479 , 0.007116 , 0.115305 , 0.044798 , 0.047459 , 0.000000 , 0.031716 , 0.184598 , 0.059302 , 0.016393],
        [0.113806 , 0.100746 , 0.033582 , 1.000000 , 0.123134 , 0.031716 , 0.000000 , 0.138060 , 0.110075 , 0.055970],
        [0.159954 , 0.000000 , 0.189808 , 0.206647 , 0.056824 , 0.184598 , 0.138060 , 0.000000 , 0.000000 , 0.000400],
        [0.044294 , 0.000000 , 0.150651 , 0.125723 , 0.044723 , 0.059302 , 0.110075 , 0.000000 , 0.000000 , 0.000000],
        [0.772091 , 0.023191 , 0.013195 , 0.044798 , 0.000000 , 0.016393 , 0.055970 , 0.000400 , 0.000000 , 0.000000]
    ])


    LGRAAL_runtimes = np.array([
        [ 0.00 , 11654.75 , 15690.76 , 11205.81 , 11855.10 , 11715.77 , 11404.67 , 11507.87 , 11847.20 , 11539.27],
        [ 11654.75 , 0.00 , 18649.59 , 11571.06 , 11837.03 , 12261.51 , 11295.99 , 11896.27 , 11988.35 , 11973.12],
        [ 15690.76 , 18649.59 , 0.00 , 12003.83 , 13085.50 , 23307.02 , 11647.24 , 12507.96 , 18777.92 , 15833.67],
        [ 11205.81 , 11571.06 , 12003.83 , 0.00 , 10976.20 , 10906.10 , 10838.40 , 10854.28 , 11193.94 , 11348.99],
        [ 11855.10 , 11837.03 , 13085.50 , 10976.20 , 0.00 , 11395.11 , 10884.13 , 10964.08 , 11097.46 , 11191.01],
        [ 11715.77 , 12261.51 , 23307.02 , 10906.10 , 11395.11 , 0.00 , 11106.61 , 11741.19 , 12138.15 , 11570.10],
        [ 11404.67 , 11295.99 , 11647.24 , 10838.40 , 10884.13 , 11106.61 , 0.00 , 10977.21 , 11636.99 , 11019.02],
        [ 11507.87 , 11896.27 , 12507.96 , 10854.28 , 10964.08 , 11741.19 , 10977.21 , 0.00 , 11371.36 , 11410.18],
        [ 11847.20 , 11988.35 , 18777.92 , 11193.94 , 11097.46 , 12138.15 , 11636.99 , 11371.36 , 0.00 , 11143.94],
        [ 11539.27 , 11973.12 , 15833.67 , 11348.99 , 11191.01 , 11570.10 , 11019.02 , 11410.18 , 11143.94 , 0.00],
    ])

    """
    Original_TAME_tri_results = np.array([
        [0.636097, 0.168017, 0.249072, 0.367052, 0.225823, 0.237254, 0.384328, 0.157939, 0.058468, 0.200720],
        [0.168017, 0.000000, 0.222244, 0.218208, 0.089656, 0.217368, 0.253731, 0.079989, 0.049944, 0.125950],
        [0.249072, 0.222244, 0.000000, 0.312139, 0.393244, 0.121830, 0.380597, 0.189237, 0.148775, 0.280288],
        [0.367052, 0.218208, 0.312139, 0.000000, 0.131503, 0.315029, 0.537313, 0.236994, 0.117052, 0.104046],
        [0.225823, 0.089656, 0.393244, 0.131503, 0.000000, 0.373566, 0.175373, 0.094602, 0.045880, 0.086765],
        [0.237254, 0.217368, 0.121830, 0.315029, 0.373566, 0.000000, 0.375000, 0.188971, 0.145701, 0.265094],
        [0.384328, 0.253731, 0.380597, 0.537313, 0.175373, 0.375000, 0.000000, 0.220149, 0.152985, 0.134328],
        [0.157939, 0.079989, 0.189237, 0.236994, 0.094602, 0.188971, 0.220149, 0.000000, 0.051225, 0.116353],
        [0.058468, 0.049944, 0.148775, 0.117052, 0.045880, 0.145701, 0.152985, 0.051225, 0.000000, 0.075170],
        [0.200720, 0.125950, 0.280288, 0.104046, 0.086765, 0.265094, 0.134328, 0.116353, 0.075170, 0.000000]
    ])
    """

    TAME_accuracy = np.array([
        [0.63609661, 0.16300982, 0.23697952, 0.36705202, 0.21109123, 0.23314896, 0.38432836, 0.12838943, 0.05461178, 0.20071971],
        [0.16300982, 0.        , 0.22026751, 0.21820809, 0.0896559 , 0.21255848, 0.25373134, 0.07998946, 0.04994399, 0.12594962],
        [0.23697952, 0.22026751, 0.        , 0.31213873, 0.39324424, 0.114944  , 0.38059701, 0.1892375 , 0.14137572, 0.28028788],
       [0.36705202, 0.21820809, 0.31213873, 0.        , 0.13150289, 0.30346821, 0.53731343, 0.23699422, 0.11560694, 0.10404624],
       [0.21109123, 0.0896559 , 0.39324424, 0.13150289, 0.        , 0.37356624, 0.17537313, 0.08723561, 0.04524887, 0.08676529],
       [0.23314896, 0.21255848, 0.114944  , 0.30346821, 0.37356624, 0.        , 0.37126866, 0.18897129, 0.13918708, 0.26509396],
       [0.38432836, 0.25373134, 0.38059701, 0.53731343, 0.17537313, 0.37126866, 0.        , 0.22014925, 0.15298507, 0.13432836],
       [0.12838943, 0.07998946, 0.1892375 , 0.23699422, 0.08723561,0.18897129, 0.22014925, 0.        , 0.0512246 , 0.11635346],
       [0.05461178, 0.04994399, 0.14137572, 0.11560694, 0.04524887, 0.13918708, 0.15298507, 0.0512246 , 0.        , 0.07516993],
       [0.20071971, 0.12594962, 0.28028788, 0.10404624, 0.08676529, 0.26509396, 0.13432836, 0.11635346, 0.07516993, 0.        ]
       ])

    """
    Original_TAME_runtimes = np.array([
        [1061.00 , 10247.21 , 46408.58 , 822.62 , 6354.54 , 36275.62 , 562.21 , 11995.29 , 16783.17 , 2067.42],
        [10247.21 , 0.00 , 31355.38 , 1675.38 , 12653.88 , 32045.00 , 1194.62 , 14261.38 , 23373.75 , 4238.50],
         [46408.58 , 31355.38 , 0.00 , 10145.12 , 26888.00 , 94027.38 , 9226.75 , 33650.62 , 42544.25 , 16983.00],
        [822.62 , 1675.38 , 10145.12 , 0.00 , 253.58 , 1565.83 , 101.88 , 329.62 , 1303.88 , 725.25],
        [6354.54 , 12653.88 , 26888.00 , 253.58 , 0.00 , 6714.00 , 338.50 , 3075.00 , 11986.12 , 4897.12],
        [36275.62 , 32045.00 , 94027.38 , 1565.83 , 6714.00 , 0.00 , 7710.12 , 29813.62 , 34031.38 , 19682.12],
        [562.21 , 1194.62 , 9226.75 , 101.88 , 338.50 , 7710.12 , 0.00 , 565.38 , 1843.88 , 263.25],
        [11995.29 , 14261.38 , 33650.62 , 329.62 , 3075.00 , 29813.62 , 565.38 , 0.00 , 14429.88 , 783.75],
        [16783.17 , 23373.75 , 42544.25 , 1303.88 , 11986.12 , 34031.38 , 1843.88 , 14429.88 , 0.00 , 11213.88],
        [2067.42 , 4238.50 , 16983.00 , 725.25 , 4897.12 , 19682.12 , 263.25 , 783.75 , 11213.88 , 0.00],
    ])
    """

    TAME_runtimes = np.array(
        [[1.05500e+03, 2.27550e+04, 1.90574e+05, 3.21400e+03, 1.29440e+04, 1.49673e+05, 2.84800e+03, 2.65410e+04, 3.94160e+04, 5.72800e+03],
        [2.27550e+04, 0.00000e+00, 7.24140e+04, 3.14800e+03, 1.59560e+04,9.83800e+04, 2.94000e+03, 1.89410e+04, 5.23860e+04, 5.05700e+03],
        [1.90574e+05, 7.24140e+04, 0.00000e+00, 7.83900e+03, 4.99760e+04,8.19281e+05, 6.56900e+03, 8.88170e+04, 1.48247e+05, 1.88770e+04],
        [3.21400e+03, 3.14800e+03, 7.83900e+03, 0.00000e+00, 2.07700e+03,1.08690e+04, 6.84000e+02, 1.60600e+03, 3.67300e+03, 1.23700e+03],
        [1.29440e+04, 1.59560e+04, 4.99760e+04, 2.07700e+03, 0.00000e+00, 4.61230e+04, 2.50600e+03, 7.02800e+03, 1.84210e+04, 3.52700e+03],
        [1.49673e+05, 9.83800e+04, 8.19281e+05, 1.08690e+04, 4.61230e+04, 0.00000e+00, 1.71970e+04, 9.63110e+04, 1.08346e+05, 2.54860e+04],
        [2.84800e+03, 2.94000e+03, 6.56900e+03, 6.84000e+02, 2.50600e+03, 1.71970e+04, 0.00000e+00, 2.56200e+03, 6.06600e+03, 1.45200e+03],
        [2.65410e+04, 1.89410e+04, 8.88170e+04, 1.60600e+03, 7.02800e+03, 9.63110e+04, 2.56200e+03, 0.00000e+00, 4.48030e+04, 3.42300e+03],
        [3.94160e+04, 5.23860e+04, 1.48247e+05, 3.67300e+03, 1.84210e+04, 1.08346e+05, 6.06600e+03, 4.48030e+04, 0.00000e+00, 9.07900e+03],
        [5.72800e+03, 5.05700e+03, 1.88770e+04, 1.23700e+03, 3.52700e+03, 2.54860e+04, 1.45200e+03, 3.42300e+03, 9.07900e+03, 0.00000e+00]
    ])

    new_TAME_tri_results = np.array([
        [0.0     , 0.164064, 0.236688, 0.299133, 0.210565, 0.23734 , 0.328358, 0.113634, 0.054039, 0.208317],
        [0.164064, 0.0     , 0.224221, 0.171965, 0.08934 , 0.214535, 0.233209, 0.081175, 0.046913, 0.12395 ],
        [0.236688, 0.224221, 0.0     , 0.40896 , 0.399663, 0.111935, 0.505597, 0.170337, 0.118447, 0.287885],
        [0.299133, 0.171965, 0.40896 , 0.0     , 0.135838, 0.384393, 0.643657, 0.15896 , 0.117052, 0.095376],
        [0.210565, 0.08934 , 0.399663, 0.135838, 0.0     , 0.376513, 0.18097 , 0.08471 , 0.047143, 0.090764],
        [0.23734 , 0.214535, 0.111935, 0.384393, 0.376513, 0.0     , 0.468284, 0.175699, 0.121834, 0.27549 ],
        [0.328358, 0.233209, 0.505597, 0.643657, 0.18097 , 0.468284, 0.0     , 0.19403 , 0.158582, 0.121269],
        [0.113634, 0.081175, 0.170337, 0.15896 , 0.08471 , 0.175699, 0.19403 , 0.0     , 0.035956, 0.112755],
        [0.054039, 0.046913, 0.118447, 0.117052, 0.047143, 0.121834, 0.158582, 0.035956, 0.0     , 0.077969],
        [0.208317, 0.12395 , 0.287885, 0.095376, 0.090764, 0.27549 , 0.121269, 0.112755, 0.077969, 0.0     ]
    ])

    new_TAME_runtimes = np.array([
        [0.0, 13.78, 209.79, 10.23, 12.13, 190.04, 10.88, 14.89, 14.07, 10.88],
        [13.78, 0.0, 214.56, 2.72, 3.9, 141.94, 2.5, 6.32, 5.22, 2.65],
        [209.79, 214.56, 0.0, 184.69, 217.84, 310.22, 210.08, 213.61, 218.12, 182.1],
        [10.23, 2.72, 184.69, 0.0, 1.23, 149.37, 0.36, 3.88, 3.25, 0.56],
        [12.13, 3.9, 217.84, 1.23, 0.0, 155.54, 1.14, 5.33, 3.87, 1.43],
        [190.04, 141.94, 310.22, 149.37, 155.54, 0.0, 151.98, 167.41, 142.73, 153.45],
        [10.88, 2.5, 210.08, 0.36, 1.14, 151.98, 0.0, 3.28, 2.94, 0.51],
        [14.89, 6.32, 213.61, 3.88, 5.33, 167.41, 3.28, 0.0, 6.32, 3.83],
        [14.07, 5.22, 218.12, 3.25, 3.87, 142.73, 2.94, 6.32, 0.0, 3.1],
        [10.88, 2.65, 182.1, 0.56, 1.43, 153.45, 0.51, 3.83, 3.1, 0.0],
    ])

    if perm is None:
        return graph_names, LGRAAL_tri_results, LGRAAL_runtimes, TAME_accuracy, \
               TAME_runtimes, new_TAME_runtimes, new_TAME_tri_results
    else:
        return [graph_names[i] for i in perm], LGRAAL_tri_results[np.ix_(perm, perm)], \
                LGRAAL_runtimes[np.ix_(perm, perm)],  TAME_accuracy[np.ix_(perm, perm)],\
                TAME_runtimes[np.ix_(perm, perm)], new_TAME_runtimes[np.ix_(perm, perm)],\
               new_TAME_tri_results[np.ix_(perm, perm)]


#TODO: consider deleting
def average_runtime_per_triangles():
    graph_names, _, LGRAAL_runtimes, _,  Original_TAME_runtimes, new_TAME_runtimes,_ = get_results()

    n,m = LGRAAL_runtimes.shape

    exp_count = 0
    LGRAAL_runtime_per_triangles = 0.0
    TAME_runtime_per_triangles = 0.0
    Lambda_TAME_runtime_per_triangles = 0.0

    for i in range(n):
        tri_count_i = triangle_counts[graph_names[i]]
        for j in range(i):
            tri_count_j = triangle_counts[graph_names[j]]
            if i == j:
                continue
            else:
                exp_count += 1
                LGRAAL_runtime_per_triangles += LGRAAL_runtimes[i,j]/(tri_count_j+ tri_count_i)
                TAME_runtime_per_triangles += Original_TAME_runtimes[i,j]/(tri_count_j+ tri_count_i)
                Lambda_TAME_runtime_per_triangles += new_TAME_runtimes[i,j]/(tri_count_j+ tri_count_i)


    return LGRAAL_runtime_per_triangles/exp_count, TAME_runtime_per_triangles/exp_count, Lambda_TAME_runtime_per_triangles/exp_count

def process_biogrid_results(data,algo="LowRankTAME"):

    truncated_params = sorted(set([y[0].split("_")[-1] for y in data[-1]]))
    param_indices = {params: i for i, params in enumerate(truncated_params)}
    exp_data = np.empty((2,len(param_indices), 30))

    if algo == "LowRankTAME":
        matched_tris,total_tris, _ ,_,profiles = data
    elif algo == "TAME":
        matched_tris,total_tris, _ ,profiles = data
    else:
        raise error("must be 'LowRankTAME' or 'TAME'")

    exp_idx = 0
    for params, profile_dict in profiles:

        ranks = profile_dict["ranks"]
        if len(ranks) < 30:
            ranks.extend([ranks[-1]] * (30 - len(ranks)))

        exp_data[exp_idx%2, param_indices[params.split("_")[-1]], :] = ranks
        exp_idx += 1

    return exp_data, param_indices

def process_synthetic_TAME_output(data):

    n_vals = {n:i for i,n in enumerate(sorted(set(data[i][3] for i in range(len(data)))))}
    p_vals = {p:i for i,p in enumerate(sorted(set(data[i][2] for i in range(len(data)))))}
    truncated_params = sorted(set([y[0].split("_")[-1] for y in data[0][-1]]))
    param_vals = {params:i for i,params in enumerate(truncated_params)}

    exp_visit_counts = {}

    trial_count = int(len(data)/(len(n_vals)*len(p_vals)))

    exp_data = -np.ones((trial_count,len(n_vals),len(p_vals), len(param_vals), 30))
    triangle_counts = -np.ones((trial_count,len(n_vals),len(p_vals), len(param_vals)))

    for (proc,seed,p_remove,n, matched_tris, total_tris, profiles) in data:
        print(matched_tris)
        print(total_tris)
        if (p_remove,n) not in  exp_visit_counts:
            exp_visit_counts[(p_remove,n)] = 0

        for params, profile_dict in profiles:

            ranks = profile_dict["ranks"]
            if len(ranks) < 30:
                ranks.extend([ranks[-1]] * (30 - len(ranks)))
            exp_data[exp_visit_counts[(p_remove,n)],n_vals[n],p_vals[p_remove],param_vals[params.split("_")[-1]], :] = ranks
            triangle_counts[exp_visit_counts[(p_remove,n)],n_vals[n],p_vals[p_remove],param_vals[params.split("_")[-1]]] = total_tris
        exp_visit_counts[(p_remove, n)] += 1
    return exp_data, n_vals, p_vals, param_vals, triangle_counts

#10/17/20 extended rank data processing
def process_synthetic_TAME_output2(data):

    n_vals = {n:i for i,n in enumerate(sorted(set(data[i][2] for i in range(len(data)))))}
    p_vals = {p:i for i,p in enumerate(sorted(set(data[i][1] for i in range(len(data)))))}
    truncated_params = sorted(set([y[0].split("_")[-1] for y in data[0][-1]]))
    param_vals = {params:i for i,params in enumerate(truncated_params)}

    exp_visit_counts = {}


    trial_count = int(len(data)/(len(n_vals)*len(p_vals)))

    exp_data = -np.ones((trial_count,len(n_vals),len(p_vals), len(param_vals), 30))
    triangle_counts = -np.ones((trial_count,len(n_vals),len(p_vals), len(param_vals)))

    for (seed,p_remove,n,acc,dw_acc,tri_match,A_tri,B_tri,max_tris,profiles) in data:
        if (p_remove,n) not in  exp_visit_counts:
            exp_visit_counts[(p_remove,n)] = 0

        for params, profile_dict in profiles:

            ranks = profile_dict["ranks"]
            if len(ranks) < 30:
                ranks.extend([ranks[-1]] * (30 - len(ranks)))
            print(exp_data.shape)
            exp_data[exp_visit_counts[(p_remove,n)],n_vals[n],p_vals[p_remove],param_vals[params.split("_")[-1]], :] = ranks
            triangle_counts[exp_visit_counts[(p_remove,n)],n_vals[n],p_vals[p_remove],param_vals[params.split("_")[-1]]] = max_tris
        exp_visit_counts[(p_remove, n)] += 1

    return exp_data, n_vals, p_vals, param_vals, triangle_counts

def process_RandomGraph_data(data):
    processed_data = {}

    for _,_,p,n,matched_tris,max_tris,profile in data:
        if p not in processed_data:
            processed_data[p] = {}

        if n not in processed_data[p]:
            processed_data[p][n] = [(matched_tris,max_tris,profile)]
        else:
            processed_data[p][n].append((matched_tris,max_tris,profile))

    return processed_data

#processed the data so output over experiment, alpha_idx, beta_idx, and ranks
def process_TAME_output2(data):

    beta_params = sorted(set([y[0].split("_")[-1] for y in data[0][-1]]))
    beta_indices = {params: i for i, params in enumerate(beta_params)}

    alpha_params = sorted(set([y[0].split("_")[0] for y in data[0][-1]]))
    alpha_indices = {params: i for i, params in enumerate(alpha_params)}

    exp_data = np.empty((len(data),len(alpha_indices),len(beta_indices),30))

    exp_idx = 0

    file_names = []

    i = 0
    for (graphA,graphB,matched_tris,total_tris,profiles) in data:


        for params,profile_dict in profiles:

            alpha_val, beta_val = params.split("_")

            ranks = profile_dict["ranks"]
            if len(ranks) < 30:
                ranks.extend([ranks[-1]]*(30-len(ranks)))

            exp_data[exp_idx,alpha_indices[alpha_val],beta_indices[beta_val],:] = ranks


        file_names.append((graphA,graphB))

        exp_idx += 1



 #       exp_data[exp_idx,:] = np.mean(exp_ranks,axis=0)

    #exps = slice(0,30)
   # excluded_files = [worm_Y2H1.ssten,"fly_Y2H1.ssten","worm_PHY1.ssten"]
    #exps = [i for i,(f_a,f_b) in enumerate(file_names) if (f_a not in excluded_files) and (f_b not in excluded_files)]


   # print(exps)
    return exp_data,beta_indices, file_names
   # return exp_data[exps,:,:], param_indices, [file_names[i] for i in exps]

def process_TAME_output(data):

    exp_data = -np.ones((len(data)*8,30))
    exp_idx = 0

    for (graphA,graphB,matched_tris,total_tris,profiles) in data:

        for _,profile_dict in profiles:
            ranks = profile_dict["ranks"]
            print(ranks)
            if len(ranks) < 30:
                ranks.extend([ranks[-1]]*(30-len(ranks)))
            exp_data[exp_idx,:] = ranks
            exp_idx += 1


    data_points = []
    for j in range(30):
        data_points.append((max(exp_data[:,j]),
                            np.percentile(exp_data[:,j], 95),
                            np.percentile(exp_data[:,j], 80),
                            np.percentile(exp_data[:,j], 50),
                            np.percentile(exp_data[:,j], 20),
                            np.percentile(exp_data[:,j], 5),
                            min(exp_data[:,j])))

    return data_points


def get_x_percent_of_data_bounds(data,x):

    data = sorted(data)
    lower_rank = ceil((1-x)/2*len(data))
    upper_rank = len(data)-lower_rank
    return data[lower_rank], data[upper_rank]

