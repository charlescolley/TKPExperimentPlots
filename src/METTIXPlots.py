from plottingStyle import * 

import matplotlib as mpl


METTcolor1 = "#D64550" # Red
#METTcolor2 = "#1B998B"  
METTcolor2 = "#06BA63" # Green
METTcolor3 = "#8C2155" # Purple 
METTcolor4 = "#446DF6" # Blue
METTcolor5 = "#E6AF2E" # Yellow
METTcolor6 = "#0E3D95" # Dark Blue
METTcolor7 = "#FA8334" # Orange  

METTFontColor ="#31572C"
METTBackGroundColor="#E2F4EDFF"
#METTBackGroundColor="#E4F2ECaa"

mpl.rcParams["figure.facecolor"] = METTBackGroundColor
mpl.rcParams["axes.facecolor"] = METTBackGroundColor
mpl.rcParams["savefig.facecolor"] = METTBackGroundColor

mpl.rcParams['text.color'] = METTFontColor
mpl.rcParams['axes.labelcolor'] = METTFontColor
mpl.rcParams["axes.edgecolor"] = METTFontColor
mpl.rcParams['xtick.color'] = METTFontColor
mpl.rcParams['ytick.color'] = METTFontColor

mpl.rcParams["grid.color"] =  METTFontColor

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Tahoma']

#
#   The precision of LR-TAME vs. TAME
#


#Shows the noise introduced by the TAME routine by considering second largest 
# singular values in the rank 1 case (alpha=1.0, beta =0.0), plots againts both 
# |V_A||V_B| and |T_A||T_B| for comparison. Data plotted for LVGNA data
def TAME_LVGNA_rank_1_case_singular_values(axes=None):
    with open(TAME_RESULTS + "Rank1SingularValues/TAME_LVGNA_iter_30_no_match_tol_1e-12.json","r") as f:
        TAME_data = json.load(f)
        TAME_data= TAME_data[-1]

    with open(TAME_RESULTS+ "Rank1SingularValues/LowRankTAME_LVGNA_iter_30_no_match_tol_1e-12.json","r") as f:
        LowRankTAME_data = json.load(f)    
        LowRankTAME_data = LowRankTAME_data[-1]


    if axes is None:
        f,axes = plt.subplots(1,1,dpi=60)
        f.set_size_inches(4, 4)

    def process_data(data):
        nonzero_second_largest_sing_vals = []
        zero_second_largest_sing_vals = []

        nonzero_vertex_products = []
        zero_vertex_products = []

        nonzero_triangle_products = []   
        zero_triangle_products = []

        for file_A,file_B,_,_,profile in data:
            graph_A =  " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B =  " ".join(file_B.split(".ssten")[0].split("_"))
            profile_dict = profile[0][-1]

            #normalize the singular values 
            for s in profile_dict["sing_vals"]:
                total = sum(s)
                s[:] = [s_i/total for s_i in s]

            #max_rank = int(max(profile_dict["ranks"]))
            #sing_vals = [s[1] if len(s) > 1 else 2e-16 for s in profile_dict["sing_vals"]
            sing_vals = [(i,s[1]) if len(s) > 1 else (i,2e-16) for (i,s) in enumerate(profile_dict["sing_vals"])]
            #find max sing val, and check iterates rank
            i,sing2_val = max(sing_vals,key=lambda x:x[1])
            rank = profile_dict["ranks"][i]

            if rank > 1:
            #if max_rank > 1.0:
                nonzero_second_largest_sing_vals.append(sing2_val)
                nonzero_vertex_products.append(vertex_counts[graph_A]*vertex_counts[graph_B])
                nonzero_triangle_products.append(triangle_counts[graph_A]*triangle_counts[graph_B])
            else:
                zero_second_largest_sing_vals.append(sing2_val)
                zero_vertex_products.append(vertex_counts[graph_A]*vertex_counts[graph_B])
                zero_triangle_products.append(triangle_counts[graph_A]*triangle_counts[graph_B])

        return nonzero_second_largest_sing_vals, nonzero_vertex_products, nonzero_triangle_products, zero_second_largest_sing_vals, zero_vertex_products, zero_triangle_products

    
    TAME_nonzero_second_largest_sing_vals, TAME_nonzero_vertex_products, \
        TAME_nonzero_triangle_products, TAME_zero_second_largest_sing_vals,\
             TAME_zero_vertex_products, TAME_zero_triangle_products = process_data(TAME_data)

    
    LowRankTAME_nonzero_second_largest_sing_vals, LowRankTAME_nonzero_vertex_products, \
        LowRankTAME_nonzero_triangle_products, LowRankTAME_zero_second_largest_sing_vals,\
             LowRankTAME_zero_vertex_products, LowRankTAME_zero_triangle_products = process_data(LowRankTAME_data)

    #
    #  Plot the data
    #

    ax = axes
    TAME_color = METTcolor1
    LRTAME_color = METTcolor4
    epsilon_color = METTcolor3
    #ax = plt.subplot(122)
    ax.set_yscale("log")
    ax.grid(which="major", axis="y")

    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=epsilon_color)
    ax.annotate("TAME", xy=(.2,.7),xycoords='axes fraction',fontsize=12,c=TAME_color)
    ax.annotate("LowRankTAME", xy=(.2,.1),xycoords='axes fraction',fontsize=12,c=LRTAME_color)
    ax.set_ylabel(r"$\sigma_2$")
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")
    #ax.set_facecolor(METTBackGroundColor)

    
    ax.set_xlim(1e5,1e12)
    ax.set_ylim(1e-16,1e-7)

    #scatter plot formatting
    marker_size = 20
    marker_alpha = 1.0

    ax.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=TAME_color,s=marker_size)
    ax.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=TAME_color,s=marker_size)

    #plot machine epsilon
    #all_triangle_products = list(itertools.chain(TAME_zero_triangle_products,TAME_nonzero_triangle_products)) 
    #plt.plot(sorted(all_triangle_products),[2e-16]*(len(all_triangle_products)),c=t3_color)
    ax.plot([1e5,1e13],[2e-16]*2,c=epsilon_color,zorder=1)

    ax.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=LRTAME_color,s=marker_size)
    scatter = ax.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=LRTAME_color,s=marker_size,zorder=2)
 
    plt.tight_layout()
    plt.show()
    

#Shows the noise introduced by the TAME routine by considering second largest 
# singular values in the rank 1 case (alpha=1.0, beta =0.0), plots againts both 
# |V_A||V_B| and |T_A||T_B| for comparison. Data plotted for RandomGeometric Graphs 
# degreedist = LogNormal(5,1). 
def TAME_RandomGeometric_rank_1_case_singular_values(axes=None):

    if axes is None:
        f,axes = plt.subplots(1,1,dpi=60)
        f.set_size_inches(3, 3)

   
    with open(TAME_RESULTS + "Rank1SingularValues/LowRankTAME_RandomGeometric_log5_iter_30_n_100_20K_no_match_tol_1e-12.json","r") as f:
        LowRankTAME_data = json.load(f)   

    with open(TAME_RESULTS + "Rank1SingularValues/TAME_RandomGeometric_degreedist:log5_alphas:[1.0]_betas:[0.0]_iter:30_trials:10_n:[1e2,5e2,1e3,2e3,5e3,1e4,2e4]_no_match_tol:1e-12.json","r") as f:
        TAME_data = json.load(f)
    

    def process_RandomGeometricResults(data):

        nonzero_second_largest_sing_vals = []
        zero_second_largest_sing_vals = []

        nonzero_vertex_products = []
        zero_vertex_products = []

        nonzero_triangle_products = []   
        zero_triangle_products = []

        n_values = set()
        for p,seed,p_remove,n,_,max_tris,profiles in data:
            n_values.add(n)
            params, profile_dict = profiles[0] #should only be alpha = 1.0, beta = 0.0

            #normalize the singular values
            for s in profile_dict["sing_vals"]: 
                s = [0.0 if x is None else x for x in s]
                #saving to json seems to introduce Nones when reading from saved Julia files
                total = sum(s)
                s[:] = [s_i/total for s_i in s]

            #max_rank = int(max(profile_dict["ranks"]))

            sing_vals = [(i,s[1]) if len(s) > 1 else (i,2e-16) for (i,s) in enumerate(profile_dict["sing_vals"])]
            #find max sing val, and check iterates rank
            i,sing2_val = max(sing_vals,key=lambda x:x[1])
            rank = profile_dict["ranks"][i]

            if rank > 1:
                #print([sum(s) for s in profile_dict["sing_vals"]])
                nonzero_second_largest_sing_vals.append(sing2_val)
                nonzero_vertex_products.append(n**2)
                nonzero_triangle_products.append(max_tris**2) 

                #TODO: need to use seed to compute actual triangle counts
            else:
                zero_second_largest_sing_vals.append(sing2_val)
                zero_vertex_products.append(n**2)
                zero_triangle_products.append(max_tris**2) 

        return n_values, nonzero_second_largest_sing_vals, nonzero_vertex_products, nonzero_triangle_products, zero_second_largest_sing_vals, zero_vertex_products, zero_triangle_products


    n_values, LowRankTAME_nonzero_second_largest_sing_vals, LowRankTAME_nonzero_vertex_products,\
         LowRankTAME_nonzero_triangle_products, LowRankTAME_zero_second_largest_sing_vals,\
              LowRankTAME_zero_vertex_products, LowRankTAME_zero_triangle_products =\
                   process_RandomGeometricResults(LowRankTAME_data)
    _, TAME_nonzero_second_largest_sing_vals, TAME_nonzero_vertex_products,\
         TAME_nonzero_triangle_products, TAME_zero_second_largest_sing_vals,\
              TAME_zero_vertex_products, TAME_zero_triangle_products =\
                   process_RandomGeometricResults(TAME_data)

    
    #
    #   Make Triangle_Triangle plots
    #
    ax = axes
    TAME_color = METTcolor1
    LRTAME_color = METTcolor4
    epsilon_color = METTcolor2

    #  --  format the axis  --  #
    ax.set_yscale("log")
    ax.grid(which="major", axis="y")

    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(3e5,7e11)
    ax.set_xticks([1e7,1e8,1e11])
    
    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=METTcolor3)
    ax.set_ylabel(r"$\sigma_2$")

    #  --  scatter plot formatting  --  #
    marker_size = 20
    marker_alpha = 1.0

    #plot the TAME Data

    #plot machine epsilon
    plt.plot([3e5,7e11],[2e-16]*2,c=METTcolor3,zorder=1)

    #plot LowRankTAME Data
    plt.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=METTcolor4, s=marker_size,zorder=2)
    plt.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=METTcolor4,s=marker_size,zorder=2)

    #plot TAME Data
    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=TAME_color,s=marker_size,zorder=3)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=TAME_color,s=marker_size,zorder=3)


    axins = ax.inset_axes([.6,.15,.25,.25]) # zoom = 6
    #axins.set_facecolor(METTBackGroundColor)
    axins.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=TAME_color,s=marker_size)
    axins.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=TAME_color,s=marker_size)
    # sub region of the original image
    #axins.set_xlim(9e9, 3e11)
    axins.set_xlim(4e9, 1e10)
    axins.set_ylim(5e-13, 1.5e-12)
    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_xticks([])
    axins.minorticks_off()
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5",alpha=.5,zorder=1)

   

    #TODO: combining plots, potentially remove later
    plt.tight_layout()
    plt.show()

#
#   LVGNA Network Alignment 
#

def make_LVGNA_runtime_performance_plots(**kwargs):

    f = plt.figure(dpi=60)
    f.set_size_inches(10, 5.5)
    #f.set_size_inches(16, 10)

    height = 0.8
    width = 0.55 
    far_left = .08
    bottom = .125
    pad = .08

    rectangle1 = [far_left, bottom, width, height]
    rectangle2 = [far_left+width+pad, bottom, .25,height]

    axes = [plt.axes(rectangle1),plt.axes(rectangle2)]

    make_LVGNA_runtime_plots(axes[0],**kwargs)
    make_LVGNA_performance_plots(axes[1],**kwargs)

    plt.show()


# TAME intro - 
#     make_LVGNA_runtime_performance_plots(algos_to_include=["TAME","LGRAAL"])
# LambdaTAME intro - 
#     make_LVGNA_runtime_performance_plots(algos_to_include=["TAME","LambdaTAME","LGRAAL"])
# LRTAME intro - 
#     make_LVGNA_runtime_performance_plots(algos_to_include=["TAME","LRTAME","LambdaTAME","LGRAAL"])
# Low Rank Matchingintro - 
#     make_LVGNA_runtime_performance_plots(algos_to_include=["TAME","LRTAME","LRTAME-lrm","LambdaTAME","LambdaTAME-rom","LGRAAL","LowRankEigenAlign"])

def make_LVGNA_performance_plots(ax=None,algos_to_include=["TAME","LRTAME","LRTAME-lrm","LambdaTAME","LambdaTAME-rom","LGRAAL","LowRankEigenAlign"]):

    graph_names, LGRAAL_tri_results, LGRAAL_runtimes, Original_TAME_tri_results, Original_TAME_runtimes, _,_ = get_results()
    def process_data(f,algorithm):
        _, exp_results = json.load(f)
        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        results = np.zeros((len(graph_names),len(graph_names)))
        runtimes = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,param_profiles) in exp_results:


            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            ratio = matched_tris/max_tris
            results[i,j] = ratio
            results[j,i] = ratio

            #sum over all params
            total_runtime = 0.0
            matching_runtime = 0.0
            for params, profile in param_profiles:
                timings = [v for k,v in profile.items() if "timings" in k]
                matching_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],[v for k,v in profile.items() if "matching_timings" in k]))
                total_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],timings))

                print(f"total_runtime: {total_runtime}  ---  matching_runtime: {matching_runtime}")
                print(total_runtime - matching_runtime)
            
            runtimes[j,i] = total_runtime
            runtimes[i,j] = total_runtime
        return runtimes, results

    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1e0,1e1,1e2]_iter_15.json", "r") as f:
        LowRankTAME_runtimes, LowRankTAME_results = process_data(f,"LowRankTAME")
        
    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1,1e1,1e2]_iter_15_low_rank_matching.json", "r") as f:
        LowRankTAME_LRMatch_runtimes, LowRankTAME_LRMatch_results = process_data(f,"LowRankTAME")

  
    with open(TAME_RESULTS + "LVGNA_Experiments/LambdaTAME_LVGNA_results_alphas:[.5,1.0]_betas:[0,1e0,1e1,1e2]_iter:15.json","r") as f:
        _, exp_results = json.load(f)
        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        LambdaTAME_results = np.zeros((len(graph_names),len(graph_names)))
        LambdaTAME_runtimes = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,profile) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            ratio = matched_tris/max_tris
            LambdaTAME_results[i,j] = ratio
            LambdaTAME_results[j,i] = ratio

            total_runtime = 0.0
            #matching_runtime = 0.0
        
            timings = [v for k,v in profile.items() if "timings" in str.lower(k)]
            #matching_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],[v for k,v in profile.items() if "matching_timings" in k]))
            total_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],timings))

            LambdaTAME_runtimes[j,i] = total_runtime
            LambdaTAME_runtimes[i,j] = total_runtime
       
    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankEigenAlign_LVGNA_iter:10.json","r") as f:
        _, exp_results = json.load(f)
        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        LowRankEigenAlign_results = np.zeros((len(graph_names),len(graph_names)))
        LowRankEigenAlign_runtimes = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = " ".join(file_A.split(".smat")[0].split("_"))
            graph_B = " ".join(file_B.split(".smat")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            ratio = matched_tris/max_tris
            LowRankEigenAlign_results[i,j] = ratio
            LowRankEigenAlign_results[j,i] = ratio

            LowRankEigenAlign_runtimes[j,i] = runtime
            LowRankEigenAlign_runtimes[i,j] = runtime

    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-12_results.json","r") as f:
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-16_profile:true_tol:1e-12_results.json","r") as f:
        exp_results = json.load(f)
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        LambdaTAMEGramMatching_results = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]
            print(file_A,file_B,matched_tris)
            ratio = matched_tris/max_tris
            LambdaTAMEGramMatching_results[j,i] = ratio
            LambdaTAMEGramMatching_results[i,j] = ratio


    n = len(graph_names)
    LGRAAL_performance = []
    LGRAAL_accuracy = []
    TAME_performance = []
    TAME_accuracy = []
    LambdaTAME_performance = []
    LambdaTAME_accuracy = []
    LambdaTAMEGramMatching_performance = []
    LambdaTAMEGramMatching_accuracy = []
    LowRankTAME_performance = []
    LowRankTAME_accuracy = []
    LowRankTAME_LRMatch_performance = []
    LowRankTAME_LRMatch_accuracy = []
    LowRankEigenAlign_performance = []
    LowRankEigenAlign_accuracy = []

    print(LambdaTAMEGramMatching_results)
    print(Original_TAME_tri_results)

    Is,Js = np.triu_indices(n,k=1)
    for i,j in zip(Is,Js):

        best = max(LambdaTAME_results[i,j],LambdaTAMEGramMatching_results[i,j],Original_TAME_tri_results[i,j],LGRAAL_tri_results[i,j],LowRankTAME_results[i,j],LowRankTAME_LRMatch_results[i,j],LowRankEigenAlign_results[i,j])

        print(f"GM - {LambdaTAMEGramMatching_results[i,j]},TAME-{Original_TAME_tri_results[i,j]}\n")

        LGRAAL_ratio = LGRAAL_tri_results[i,j]/best
        #LGRAAL_accuracy.append(LGRAAL_tri_results[i,j])
        LGRAAL_performance.append(LGRAAL_ratio)

        #TODO: refactor to TAME
        ratio =  Original_TAME_tri_results[i,j]/ best
        #TAME_accuracy.append(Original_TAME_tri_results[i,j])
        TAME_performance.append(ratio)

        #TODO: refactor to LambdaTAME
       # LambdaTAME_accuracy.append(LambdaTAME_results[i,j])
        ratio = LambdaTAME_results[i, j] / best
        LambdaTAME_performance.append(ratio)

        ratio = LambdaTAMEGramMatching_results[i,j] / best
        LambdaTAMEGramMatching_performance.append(ratio)

      #  LowRankTAME_accuracy.append(LowRankTAME_results[i,j])
        ratio = LowRankTAME_results[i, j] / best
        LowRankTAME_performance.append(ratio)

     #   LowRankTAME_LRMatch_accuracy.append(LowRankTAME_LRMatch_results[i,j])
        ratio = LowRankTAME_LRMatch_results[i, j] / best
        LowRankTAME_LRMatch_performance.append(ratio)

        ratio = LowRankEigenAlign_results[i,j] /best
        LowRankEigenAlign_performance.append(ratio)
    
    #LGRAAL_accuracy      = sorted(LGRAAL_accuracy,reverse=True)
    LGRAAL_performance   = sorted(LGRAAL_performance,reverse=True)
    TAME_performance = sorted(TAME_performance,reverse=True)
    #TAME_accuracy    = sorted(TAME_accuracy ,reverse=True)
    LambdaTAME_performance = sorted(LambdaTAME_performance,reverse=True)
    
    LambdaTAMEGramMatching_performance = sorted(LambdaTAMEGramMatching_performance,reverse=True)
    #LambdaTAME_accuracy    = sorted(LambdaTAME_accuracy,reverse=True)
    LowRankTAME_performance = sorted(LowRankTAME_performance,reverse=True)
    #LowRankTAME_accuracy    = sorted(LowRankTAME_accuracy,reverse=True)
    LowRankTAME_LRMatch_performance = sorted(LowRankTAME_LRMatch_performance,reverse=True)
    #LowRankTAME_LRMatch_accuracy    = sorted(LowRankTAME_LRMatch_accuracy,reverse=True)
    LowRankEigenAlign_performance = sorted(LowRankEigenAlign_performance,reverse=True)

    if ax is None:
        fig, ax = plt.subplots(1,1)
        show_plot = True
    else:
        show_plot = False

    

    print(algos_to_include)
    if "TAME" in algos_to_include:
        ax.plot(range(len(TAME_performance)), TAME_performance, label="TAME", c=METTcolor1)
        #bold_outlined_text(ax,"TAME (C++)",t1_color,(.82, .9))
        if len(algos_to_include) == 2:
            ax.annotate("TAME (C++)",xy=(.55, .875), xycoords='axes fraction', c=METTcolor1,rotation=-10)
        elif len(algos_to_include) == 3 or len(algos_to_include) == 4:
            ax.annotate("TAME (C++)",xy=(.425, .81), xycoords='axes fraction', c=METTcolor1,rotation=-10)
        else:
            ax.annotate("TAME (C++)",xy=(.59, .81), xycoords='axes fraction', c=METTcolor1,rotation=-18)
   
    if "LambdaTAME-rom" in algos_to_include:
        ax.plot(range(len(LambdaTAME_performance)), LambdaTAME_performance, label="$\Lambda$-TAME - (rom)", c=darkest_t2_color,linestyle=(0,(10,1.5)))
        ax.annotate("$\Lambda$-TAME-(rom)",xy=(.4, .77), xycoords='axes fraction', c=darkest_t2_color,rotation=-27.5)
        
    if "LambdaTAME" in algos_to_include:
        ax.plot(range(len(LambdaTAMEGramMatching_performance)), LambdaTAMEGramMatching_performance, label="$\Lambda$-TAME", c=METTcolor2,linestyle="--")
        ax.annotate("$\Lambda$-TAME",xy=(.4, .96), xycoords='axes fraction', c=METTcolor2,rotation=0)
        
    if "LRTAME" in algos_to_include:
        ax.plot(range(len(LowRankTAME_performance)), LowRankTAME_performance, label="LowRankTAME", c=METTcolor4,linestyle=(0, (3, 1, 1, 1)))
        ax.annotate("LowRankTAME",xy=(.65, .91), xycoords='axes fraction', c=METTcolor4)

    if "LRTAME-lrm" in algos_to_include:
        ax.plot(range(len(LowRankTAME_LRMatch_performance)), LowRankTAME_LRMatch_performance, label="LowRankTAME", c=METTcolor6,linestyle="-.")
        ax.annotate("LowRankTAME-(lrm)",xy=(.3, .7),xycoords='axes fraction', c=METTcolor6,rotation=-25)
    
    if "LowRankEigenAlign" in algos_to_include:
        ax.plot(range(len(LowRankEigenAlign_performance)),LowRankEigenAlign_performance,label="LowRankEigenAlign", c=METTcolor3,linestyle=":")
        ax.annotate("LowRankEigenAlign",xy=(.1, .175),xycoords='axes fraction', c=METTcolor3,rotation=-32.5)
    
    if "LGRAAL" in algos_to_include:
        ax.plot(range(len(LGRAAL_performance)), LGRAAL_performance, label="LGRAAL", c=METTcolor5,linestyle=(0,(3,1,1,1,1,1)))
        ax.annotate("L-GRAAL",xy=(.47, .46), xycoords='axes fraction', c=METTcolor5)
        
    
    ax.set_ylabel("performance ratio")
    ax.grid(which="both")
    ax.set_xlabel("experiment order")

    if show_plot:
        plt.show()

def make_LVGNA_runtime_plots(ax=None,algos_to_include=["TAME","LRTAME","LRTAME-lrm","LambdaTAME","LambdaTAME-rom","LGRAAL","LowRankEigenAlign"]):

    graph_names, LGRAAL_tri_results, LGRAAL_runtimes, \
        Original_TAME_tri_results, TAME_runtimes, LambdaTAME_runtimes,\
             new_TAME_tri_results = get_results()
    

    def process_LowRankTAME_data(f):
        _, results = json.load(f)

        #
        #  Parse Input files
        #

        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}
        runtimes = np.zeros((len(graph_names),len(graph_names)))
        problem_sizes = []

        for (file_A,file_B,matched_tris,max_tris,param_profiles) in results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            #sum over all params
            total_runtime = 0.0
            matching_runtime = 0.0
            for params, profile in param_profiles:
                timings = [v for k,v in profile.items() if "timings" in k]
                matching_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],[v for k,v in profile.items() if "matching_timings" in k]))
                total_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],timings))

                print(f"total_runtime: {total_runtime}  ---  matching_runtime: {matching_runtime}")
                print(total_runtime - matching_runtime)
            
            runtimes[i,j] = total_runtime
            runtimes[j,i] = total_runtime

        return runtimes 


    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1e0,1e1,1e2]_iter_15.json", "r") as f:
        LowRankTAME_runtimes = process_LowRankTAME_data(f)
        
    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1,1e1,1e2]_iter_15_low_rank_matching.json", "r") as f:
        LowRankTAME_LRM_runtimes = process_LowRankTAME_data(f)
        

    with open(TAME_RESULTS + "LVGNA_Experiments/LambdaTAME_LVGNA_results_alphas:[.5,1.0]_betas:[0,1e0,1e1,1e2]_iter:15.json","r") as f:
        _, exp_results = json.load(f)
        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        LambdaTAME_runtimes = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,profile) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            total_runtime = 0.0
            #matching_runtime = 0.0
        
            timings = [v for k,v in profile.items() if "timings" in str.lower(k)]
            #matching_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],[v for k,v in profile.items() if "matching_timings" in k]))
            total_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],timings))

            LambdaTAME_runtimes[j,i] = total_runtime
            LambdaTAME_runtimes[i,j] = total_runtime

    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankEigenAlign_LVGNA_iter:10.json","r") as f:
        _, exp_results = json.load(f)
        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}


        LowRankEigenAlign_runtimes = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = " ".join(file_A.split(".smat")[0].split("_"))
            graph_B = " ".join(file_B.split(".smat")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            LowRankEigenAlign_runtimes[j,i] = runtime
            LowRankEigenAlign_runtimes[i,j] = runtime

    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-12_results.json","r") as f:
    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-16_profile:true_tol:1e-12_results.json","r") as f:
        exp_results = json.load(f)
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        LambdaTAMEGramMatching_runtimes = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            runtime = sum([sum(rt) for rt in runtime.values()])
            LambdaTAMEGramMatching_runtimes[j,i] = runtime
            LambdaTAMEGramMatching_runtimes[i,j] = runtime

    n = len(graph_names)
    problem_sizes = []

    LGRAAL_exp_runtimes = []
    TAME_exp_runtimes = []
    LambdaTAME_exp_runtimes = []
    LambdaTAMEGramMatching_exp_runtimes = []
    LowRankTAME_exp_runtimes = []
    LowRankTAME_LRM_exp_runtimes = []
    LowRankEigenAlign_exp_runtimes = []

    Is,Js = np.triu_indices(n,k=1)
    for i,j in zip(Is,Js):

        LGRAAL_exp_runtimes.append(LGRAAL_runtimes[i,j])
        TAME_exp_runtimes.append(TAME_runtimes[i,j])
        LambdaTAME_exp_runtimes.append(LambdaTAME_runtimes[i,j])
        LambdaTAMEGramMatching_exp_runtimes.append(LambdaTAMEGramMatching_runtimes[i,j])
        LowRankTAME_exp_runtimes.append(LowRankTAME_runtimes[i, j])
        LowRankTAME_LRM_exp_runtimes.append(LowRankTAME_LRM_runtimes[i, j])
        LowRankEigenAlign_exp_runtimes.append(LowRankEigenAlign_runtimes[i,j])

        problem_sizes.append(triangle_counts[graph_names[i]]*triangle_counts[graph_names[j]])

    #
    #  Plot results
    #


    if ax is None:
        fig, ax = plt.subplots(1,1)
        show_plot = True
    else:
        show_plot = False
    #ax = [ax] #jerry rigged

    ax.set_ylim(1e0,2e6)
    ax.set_xlim(2e5,5e11)

    ax.set_ylabel("runtime (s)")
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_xscale("log")
    ax.set_yscale("log")
    loess_smoothing_frac = .3
    ax.grid(which="major",zorder=-2)
    
    if "TAME" in algos_to_include:
        ax.scatter(problem_sizes,TAME_exp_runtimes,label="TAME", c=METTcolor1,marker='s')
        plot_1d_loess_smoothing(problem_sizes,TAME_exp_runtimes,loess_smoothing_frac,ax,c=METTcolor1,linestyle="solid")
        #ax[0].plot(range(len(old_TAME_performance)), old_TAME_performance, label="TAME", c=t4_color)
        ax.annotate("TAME (C++)",xy=(.53, .85), xycoords='axes fraction', c=METTcolor1)

    
    if "LambdaTAME-rom" in algos_to_include:
        ax.scatter(problem_sizes,LambdaTAME_exp_runtimes,label="$\Lambda$-TAME-(rom)",facecolors='none', edgecolors=darkest_t2_color,marker='^')
        plot_1d_loess_smoothing(problem_sizes,LambdaTAME_exp_runtimes,loess_smoothing_frac,ax,c=darkest_t2_color,linestyle=(0,(10,1.5)))
        ax.annotate("$\Lambda$-TAME - (rom)",xy=(.65, .17), xycoords='axes fraction', c=darkest_t2_color)
        
    if "LambdaTAME" in algos_to_include:
        ax.scatter(problem_sizes,LambdaTAMEGramMatching_exp_runtimes,label="$\Lambda$-TAME", c=METTcolor2 ,marker='^')
        plot_1d_loess_smoothing(problem_sizes,LambdaTAMEGramMatching_exp_runtimes,loess_smoothing_frac,ax,c=METTcolor2 ,linestyle="--")
        ax.annotate("$\Lambda$-TAME",xy=(.81, .54), xycoords='axes fraction', c=METTcolor2)

    if "LRTAME" in algos_to_include:
        ax.scatter(problem_sizes,LowRankTAME_exp_runtimes,label="LowRankTAME", c=METTcolor4)
        plot_1d_loess_smoothing(problem_sizes,LowRankTAME_exp_runtimes,loess_smoothing_frac,ax,c=METTcolor4,linestyle=(0, (3, 1, 1, 1)))
        ax.annotate("LowRankTAME",xy=(.81, .795), xycoords='axes fraction', c=METTcolor4)
    
    if "LRTAME-lrm" in algos_to_include:
        ax.scatter(problem_sizes,LowRankTAME_LRM_exp_runtimes,facecolors='none',edgecolors=METTcolor6,label="LowRankTAME-(lrm)")
        plot_1d_loess_smoothing(problem_sizes,LowRankTAME_LRM_exp_runtimes,loess_smoothing_frac,ax,c=METTcolor6,linestyle="-.")
        ax.annotate("LowRankTAME-(lrm)",xy=(.74, .48), xycoords='axes fraction', c=METTcolor6)
    
    if "LowRankEigenAlign" in algos_to_include:
        ax.scatter(problem_sizes,LowRankEigenAlign_exp_runtimes,label="LowRankEigenAlign",c=METTcolor3,marker="*")
        plot_1d_loess_smoothing(problem_sizes,LowRankEigenAlign_exp_runtimes,loess_smoothing_frac,ax,c=METTcolor3,linestyle=":")
        ax.annotate("LowRankEigenAlign",xy=(.08, .26), xycoords='axes fraction', c=METTcolor3)
        
    if "LGRAAL" in algos_to_include:
        ax.scatter(problem_sizes, LGRAAL_exp_runtimes,label="$LGRAAL", c=METTcolor5,zorder=-1,marker='x')
        plot_1d_loess_smoothing(problem_sizes,LGRAAL_exp_runtimes,loess_smoothing_frac,ax,c=METTcolor5,linestyle=(0,(3,1,1,1,1,1)))
        ax.annotate("L-GRAAL",xy=(.2, .7), xycoords='axes fraction', c=METTcolor5)


def make_LVGNA_TTVMatchingRatio_runtime_plots(ax=None):
    graph_names,TAME_BM_runtimes,TAME_impTTV_runtimes  = get_TAME_LVGNA_runtimes()

    TAME_BM_ratio = np.divide(TAME_BM_runtimes,TAME_impTTV_runtimes)

    def process_LowRankTAME_data(f):
        _, results = json.load(f)

        #
        #  Parse Input files
        #

        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}
        matchingRuntimes = np.zeros((len(graph_names),len(graph_names)))
        contractionRuntimes = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,param_profiles) in results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            #sum over all params
            totalContractionRuntime = 0.0
            contraction_times = ['qr_timings', 'contraction_timings','svd_timings']
            totalMatchingRuntime = 0.0

            for params, profile in param_profiles:

                contraction_timings = [v for k,v in profile.items() if k in contraction_times]
                totalContractionRuntime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],contraction_timings))
                totalMatchingRuntime += sum(profile['matching_timings'])
            
            contractionRuntimes[i,j] = totalContractionRuntime
            contractionRuntimes[j,i] = totalContractionRuntime
            matchingRuntimes[i,j] = totalMatchingRuntime
            matchingRuntimes[j,i] = totalMatchingRuntime

        return matchingRuntimes, contractionRuntimes


    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1e0,1e1,1e2]_iter_15.json", "r") as f:
        matchingRuntimes, contractionRuntimes = process_LowRankTAME_data(f)
        LowRankTAME_ratio = np.divide(matchingRuntimes,contractionRuntimes)
       
        
    with open(TAME_RESULTS + "LVGNA_Experiments/LowRankTAME_LVGNA_alpha_[.5,1.0]_beta_[0,1,1e1,1e2]_iter_15_low_rank_matching.json", "r") as f:
        matchingRuntimes,contractionRuntimes = process_LowRankTAME_data(f)
        LowRankTAME_LRM_ratio = np.divide(matchingRuntimes,contractionRuntimes)
        

    with open(TAME_RESULTS + "LVGNA_Experiments/LambdaTAME_LVGNA_results_alphas:[.5,1.0]_betas:[0,1e0,1e1,1e2]_iter:15.json","r") as f:
        _, exp_results = json.load(f)
        #Format the data to 
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        LambdaTAME_ratio = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,profile) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            matching_rt = 0.0
            #matching_runtime = 0.0
        
            #timings = [v for k,v in profile.items() if 'Krylov Timings' in str.lower(k)]
            #matching_runtime += sum(reduce(lambda l1,l2: [x + y for x,y in zip(l1,l2)],[v for k,v in profile.items() if "matching_timings" in k]))
            matching_rt = sum(profile["Krylov Timings"])
            contraction_rt = sum(profile['TAME_timings'])

            LambdaTAME_ratio[j,i] = matching_rt/contraction_rt 
            LambdaTAME_ratio[i,j] = matching_rt/contraction_rt 
  
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-12_results.json","r") as f:  
    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-16_profile:true_tol:1e-12_results.json","r") as f:
        exp_results = json.load(f)
        exp_idx = {name:i for i,name in enumerate(graph_names)}

        LambdaTAMEGramMatching_ratio = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            contraction_rt = sum(runtime['TAME_timings'])
            matching_rt = sum(runtime['Matching Timings'])

            LambdaTAMEGramMatching_ratio[j,i] = matching_rt/contraction_rt
            LambdaTAMEGramMatching_ratio[i,j] = matching_rt/contraction_rt

    n = len(graph_names)
    problem_sizes = []

    TAME_exp_ratios = []
    LambdaTAME_exp_ratios = []
    LambdaTAMEGramMatching_exp_ratios = []
    LowRankTAME_exp_ratios = []
    LowRankTAME_LRM_exp_ratios = []

    Is,Js = np.triu_indices(n,k=1)
    for i,j in zip(Is,Js):

        TAME_exp_ratios.append(TAME_BM_ratio[i,j])
        LambdaTAME_exp_ratios.append(LambdaTAME_ratio[i,j])
        LambdaTAMEGramMatching_exp_ratios.append(LambdaTAMEGramMatching_ratio[i,j])
        LowRankTAME_exp_ratios.append(LowRankTAME_ratio[i, j])
        LowRankTAME_LRM_exp_ratios.append(LowRankTAME_LRM_ratio[i, j])
        problem_sizes.append(triangle_counts[graph_names[i]]*triangle_counts[graph_names[j]])

    #return problem_sizes, TAME_exp_ratios

    #
    #  Plot results
    #


    if ax is None:
        fig, ax = plt.subplots(1,1)
        show_plot = True
    else:
        show_plot = False
    #ax = [ax] #jerry rigged

    ax.set_ylim(1e-3,1e5)
    ax.set_xlim(2e5,5e11)

    ax.set_ylabel("matching / contraction runtime ratio")
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    #left axis labels
  
    ax.set_xscale("log")
    ax.set_yscale("log")
    loess_smoothing_frac = .3
    ax.grid(which="major",zorder=-2)
    ax.axhspan(1e-5,1,alpha=.1,color=METTFontColor)
    
    print(TAME_exp_ratios)
    ax.scatter(problem_sizes,TAME_exp_ratios,label="TAME", c=METTcolor1,marker='s')
    plot_1d_loess_smoothing(problem_sizes,TAME_exp_ratios,loess_smoothing_frac,ax,c=METTcolor1,linestyle="solid",logFilter=True)
    #ax[0].plot(range(len(old_TAME_performance)), old_TAME_performance, label="TAME", c=t4_color)
    ax.annotate("TAME (C++)",xy=(.8, .175), xycoords='axes fraction', c=METTcolor1)

 
    ax.scatter(problem_sizes,LowRankTAME_exp_ratios,label="LowRankTAME", c=METTcolor4)
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_exp_ratios,loess_smoothing_frac,ax,c=METTcolor4,linestyle=(0, (3, 1, 1, 1)),logFilter=True)
    ax.annotate("LowRankTAME",xy=(.75, .66), xycoords='axes fraction', c=METTcolor4)
 
    ax.scatter(problem_sizes,LowRankTAME_LRM_exp_ratios,facecolors='none',edgecolors=METTcolor6,label="LowRankTAME-(lrm)")
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_LRM_exp_ratios,loess_smoothing_frac,ax,c=METTcolor6,linestyle="-.",logFilter=True)
    ax.annotate("LowRankTAME-(lrm)",xy=(.7, .03), xycoords='axes fraction', c=METTcolor6)
     
  
    ax.scatter(problem_sizes,LambdaTAMEGramMatching_exp_ratios,label="$\Lambda$-TAME", c=METTcolor2 ,marker='^')
    plot_1d_loess_smoothing(problem_sizes,LambdaTAMEGramMatching_exp_ratios,loess_smoothing_frac,ax,c=METTcolor2 ,linestyle="--")
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    ax.annotate("$\Lambda$-TAME",xy=(.85, .8), xycoords='axes fraction', c=METTcolor2)

    #print(new_TAME_exp_runtimes)
    ax.scatter(problem_sizes,LambdaTAME_exp_ratios,label="$\Lambda$-TAME-(rom)",facecolors='none', edgecolors=darkest_t2_color,marker='^')
    plot_1d_loess_smoothing(problem_sizes,LambdaTAME_exp_ratios,loess_smoothing_frac,ax,c=darkest_t2_color,linestyle=(0,(10,1.5)))
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    ax.annotate("$\Lambda$-TAME\n(rom)",xy=(.875, .45), xycoords='axes fraction', ha="center",c=darkest_t2_color)
    

    #ax.text(.95, .6,"contraction\ndominates")#, ha="center",c="k",rotation=90)
    #ax.text(.95, .1,"contraction\ndominates")#, ha="center",c="k",rotation=90)

    ax.annotate("matching\ndominates",xy=(1.05, .6), xycoords='axes fraction', ha="center",c=METTFontColor,rotation=90)
    ax.annotate("contraction\ndominates",xy=(1.05, .1), xycoords='axes fraction', ha="center",c=METTFontColor,rotation=90)

    plt.show()

#
#   Maximum Rank Experiments
#

def max_rank_LVGNA_data(ax):
    #with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_LVGNA.json", 'r') as f:
    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_RandomGeometric_degreedist:logNormalLog5_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_noiseModel:DuplicationNoise_nSizes:[100,500,1000,2000,5000,10000],p=[.5,1.0]_tol:1e-12_trials:50__maxRankTrialResults.json","r") as f:
        MM_results  = json.load(f)
    MM_exp_data, param_indices,MM_file_names = process_TAME_output2(MM_results[1])
    """
    included_exps = [sorted(("human_Y2H1.ssten","yeast_Y2H1.ssten")),
                     sorted(("human_PHY1.ssten","yeast_PHY1.ssten")),
                     sorted(("human_PHY2.ssten","yeast_PHY2.ssten"))]

    exps = [i for i,files in enumerate(MM_file_names) if sorted(files) in included_exps]
    print(exps)
    MM_file_names = [MM_file_names[i] for i in exps]
    print(MM_file_names)
    
    MM_exp_data = MM_exp_data[exps,:,:]
    """
    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_Biogrid.json","r") as f:
        Biogrid_results = json.load(f)

    Biogrid_exp_data, _ = process_biogrid_results(Biogrid_results)
    label_meta = [
        ((.05, .35),"o",METTcolor1),
        ((.05, .59),"v",METTcolor2),
        ((.1, .8),"*",METTcolor3),
        ((.09, .1),"s",METTcolor4)]

    for (param, j),(loc,marker,c) in zip(param_indices.items(),label_meta):
        n_points = []
        max_ranks = []
        ax.annotate(param, xy=loc, xycoords='axes fraction',c=c)

        for i in range(MM_exp_data.shape[0]):

            graph_names = [str.join(" ", x.split(".ssten")[0].split("_")) for x in MM_file_names[i]]
            avg_n = np.mean([vertex_counts[f] for f in graph_names])

            tri_counts = triangle_counts[graph_names[0]]*triangle_counts[graph_names[1]]
           # tri_counts = sum([triangle_counts[graph_names[0]] for f in graph_names])
            n_points.append(tri_counts)
            max_ranks.append(np.max(MM_exp_data[i,:,j,:]))
            if param == "β:100.0":
                if np.max(MM_exp_data[i,:,j,:]) < np.max(MM_exp_data[i,:,:,:]):
   
                    print(graph_names)
                    print(tri_counts)
                    print(np.max(MM_exp_data[:,:,j,:]))
                    print(np.max(MM_exp_data[i,:,j,:]))
            
        n_points.append(407650*347079)
        max_ranks.append(np.max(Biogrid_exp_data[:,j,:]))


        ax.scatter(n_points,max_ranks,c=c,s=15,alpha=.4,marker=marker)
        plot_1d_loess_smoothing(n_points,max_ranks,.3,ax,c=c,linestyle="--")
    #     plt.plot(xout,yout,c=c,linestyle="--")
    
    ax.set_xscale("log")
    ax.set_title("LVGNA")
    ax.set_xlabel(r"$|T_A||T_B|$")
    ax.yaxis.set_ticks_position('both')
    ax.set_yticks([50,100,150])
    ax.set_xlim(1e5,1e12)
    ax.set_ylabel("maximum rank")
    ax.tick_params(labeltop=False, labelright=True)
    
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.set_xticks([1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    ax.grid(which="major", axis="y")


def max_rank_synthetic_data(ax):
    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_RandomGeometric_degreedist_log5_iter:15_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100,500,1K,2K,5K,10K,20K]_noMatching_pRemove:[.01,.05]_tol:1e-12_trialcount:50.json","r") as f:
        synth_results = json.load(f)
    
    MM_exp_data, n_vals, p_vals, param_vals, tri_counts = process_synthetic_TAME_output2(synth_results)
 
    #ax = axes[0]
    ax.set_xscale("log")
    ax.set_ylim(00,315)

    label_meta = [
        ((.65, .07),"o",METTcolor1),
        ((.65, .22),"v",METTcolor2),
        ((.65, .5),"*",METTcolor3),
        ((.75, .85),"s",METTcolor4)]#  no_ylim_points
    #label_meta = [((.92, .25),t1_color),((.8, .51),t2_color),((.625, .8),t3_color),((.25, .88),t4_color)]

    #beta=100 annotations
    beta_100_meta = [(.15,.4),(.42,1.01),(.51,1.01),(.6,1.01),(.725,1.01),(.8,1.01),(.9,1.01)]

    #beta=10 annotations
    beta_10_meta = [(.15,.32),(.42,.515),(.51,.6),(.6,.65),(.725,.75),(.8,.95),(.9,.95)]

    for (params,k),(loc,marker,c) in zip(param_vals.items(),label_meta):
        
        max_vals = []
        mean_tris = []
        
        for n,j in n_vals.items():
            max_vals.append(np.max(MM_exp_data[:,j,:,k,:]))
            mean_tris.append(np.mean(tri_counts[:,j,:,k]))
        

        ax.plot(mean_tris,max_vals,c=c)
        ax.annotate(params, xy=loc, xycoords='axes fraction',c=c)


    ax.set_title("Synthetic graphs")
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=True)
    ax.grid(which="major", axis="y")
    

    ax.set_xticklabels([1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    ax.set_xlim(1e6,5e11)
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.set_ylabel("maximum rank")
    ax.set_yticks([50,100,150,200,250,300])
    

def max_rank_experiments():

    #f = plt.figure(dpi=60)
    #f.set_size_inches(10, 4)
    f, axes = plt.subplots(1,2,figsize=(10, 4),dpi=60)

    max_rank_synthetic_data(axes[0])
    max_rank_LVGNA_data(axes[1])
    
    plt.tight_layout()
    plt.show()

def max_rank_experiments_v2(plotRankRatio=False):


    f = plt.figure(dpi=60)
    #f.set_size_inches(5, 3.5)
    #f, axes = plt.subplots(2,1)
    
    f, axes = plt.subplots(1,2)
    f.set_size_inches(9, 3.5)


    shift1_color = METTcolor1
    shift2_color = METTcolor2
    shift3_color = METTcolor7
    shift4_color = METTcolor4
    #-------------------------------Plot Synth Data--------------------------------------

    #with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_RandomGeometric_degreedist_log5_iter:15_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100,500,1K,2K,5K,10K,20K]_noMatching_pRemove:[.01,.05]_tol:1e-12_trialcount:50.json","r") as f:
    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_RandomGeometric_degreedist:logNormalLog5_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_noiseModel:DuplicationNoise_nSizes:[100,500,1000,2000,5000,10000],p=[.5,1.0]_tol:1e-12_trials:50__maxRankTrialResults.json","r") as f:
        synth_results = json.load(f)

    MM_exp_data, n_vals, p_vals, param_vals, tri_counts = process_synthetic_TAME_output2(synth_results)
 
    #return MM_exp_data, n_vals, p_vals, param_vals

    ax = axes[0]
    ax.set_xscale("log")
    
    if plotRankRatio:
        label_meta = [
            ((.01, .3),"o",shift1_color),
            ((.01, .51),"v",shift2_color),
            ((.00, .65),"*",shift3_color),
            ((.01, .85),"s",shift4_color)]#  no_ylim_points
    else:
        label_meta = [
            ((.85, .07),"o",shift1_color),
            ((.85, .22),"v",shift2_color),
            ((.7, .425),"*",shift3_color),
            ((.83, .85),"s",shift4_color)]#  no_ylim_points
    #label_meta = [((.92, .25),t1_color),((.8, .51),t2_color),((.625, .8),t3_color),((.25, .88),t4_color)]

    #beta=100 annotations
    beta_100_meta = [(.15,.4),(.42,1.01),(.51,1.01),(.6,1.01),(.725,1.01),(.8,1.01),(.9,1.01)]

    #beta=10 annotations
    beta_10_meta = [(.15,.32),(.42,.515),(.51,.6),(.6,.65),(.725,.75),(.8,.95),(.9,.95)]

    for (params,k),(loc,marker,c) in zip(param_vals.items(),label_meta):
        
        max_vals = []
        mean_tris = []
        
        for n,j in n_vals.items():
            if plotRankRatio:
                max_vals.append(np.max(MM_exp_data[:,j,:,k,:])/n)
            else:
                max_vals.append(np.max(MM_exp_data[:,j,:,k,:]))
            mean_tris.append(np.mean(tri_counts[:,j,:,k]))
        

        ax.plot(mean_tris,max_vals,c=c)

        ax.annotate(params, xy=loc, xycoords='axes fraction',c=c)
        #ax.plot(marker_loc[0],marker_loc[1],marker,c=c,markersize=5)


    #ax.set_xticks(list(n_vals.keys()))
    ax.set_title("Synthetic graphs")
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=True)
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    #ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.grid(which="major", axis="y")
    #ax.set_xlim(3e6,2e11)

    ax.set_xticklabels([1e6,1e7,1e8,1e9,1e10,1e11])
    ax.set_xlim(1e6,1e11)
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())

    ax.set_xlabel(r"$|T_A||T_B|$")
    if plotRankRatio:
        ax.set_yticks(list(np.linspace(0.0,1.0,5)))
        ax.set_ylabel("maximum rank / min(n,m)")
        ax.set_ylim(0.0,1.0)
    else:
        ax.set_ylim(00,300)
        ax.set_yticks([50,100,150,200,250,300])
        ax.set_ylabel("maximum rank")
    #ax.set_xticks([5e6,1e7,1e8,1e9,1e10,1e11])
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    
    """---------------------------Plot Real World Data--------------------------------"""
    ax = axes[1]
    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_LVGNA.json", 'r') as f:
        MM_results  = json.load(f)
    MM_exp_data, param_indices,MM_file_names = process_TAME_output2(MM_results[1])
    """
    included_exps = [sorted(("human_Y2H1.ssten","yeast_Y2H1.ssten")),
                     sorted(("human_PHY1.ssten","yeast_PHY1.ssten")),
                     sorted(("human_PHY2.ssten","yeast_PHY2.ssten"))]

    exps = [i for i,files in enumerate(MM_file_names) if sorted(files) in included_exps]
    print(exps)
    MM_file_names = [MM_file_names[i] for i in exps]
    print(MM_file_names)
    
    MM_exp_data = MM_exp_data[exps,:,:]
    """
    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_Biogrid.json","r") as f:
        Biogrid_results = json.load(f)

    Biogrid_exp_data, _ = process_biogrid_results(Biogrid_results)
    if plotRankRatio:
        label_meta = [
        ((.01, .45),"o",shift1_color),
        ((.01, .65),"v",shift2_color),
        ((.1, .9),"*",shift3_color),
        ((.09, .1),"s",shift4_color)]
    else:
        label_meta = [
            ((.05, .35),"o",shift1_color),
            ((.05, .59),"v",shift2_color),
            ((.1, .8),"*",shift3_color),
            ((.09, .1),"s",shift4_color)]

    for (param, j),(loc,marker,c) in zip(param_indices.items(),label_meta):
        n_points = []
        max_ranks = []
        ax.annotate(param, xy=loc, xycoords='axes fraction',c=c)

        for i in range(MM_exp_data.shape[0]):

            graph_names = [str.join(" ", x.split(".ssten")[0].split("_")) for x in MM_file_names[i]]
            print(graph_names)
            graphA, graphB = graph_names
            avg_n = np.mean([vertex_counts[f] for f in graph_names])

            
            tri_counts = triangle_counts[graphA]*triangle_counts[graphB]
            # tri_counts = sum([triangle_counts[graph_names[0]] for f in graph_names])
            n_points.append(tri_counts)
            if plotRankRatio:
                max_ranks.append(np.max(MM_exp_data[i,:,j,:])/(min(vertex_counts[graphA],vertex_counts[graphB])))
            else:
                max_ranks.append(np.max(MM_exp_data[i,:,j,:]))
            
            if param == "β:100.0":
                if np.max(MM_exp_data[i,:,j,:]) < np.max(MM_exp_data[i,:,:,:]):
   
                    print(graph_names)
                    print(tri_counts)
                    print(np.max(MM_exp_data[:,:,j,:]))
                    print(np.max(MM_exp_data[i,:,j,:]))
            
        n_points.append(407650*347079)
        if plotRankRatio:
            max_ranks.append(np.max(Biogrid_exp_data[:,j,:])/min(5850,79458))
        else:
            max_ranks.append(np.max(Biogrid_exp_data[:,j,:]))


        ax.scatter(n_points,max_ranks,c=c,s=15,alpha=.4,marker=marker)
        plot_1d_loess_smoothing(n_points,max_ranks,.3,ax,c=c,linestyle="--",logFilter=True)
        #plt.plot(xout,yout,c=c,linestyle="--")

    ax.set_xscale("log")
    ax.set_title("LVGNA")
    ax.set_xlabel(r"$|T_A||T_B|$")
    ax.yaxis.set_ticks_position('both')
    if plotRankRatio:
        ax.set_yticks(np.linspace(0.0,.05,6))
        ax.set_ylabel("maximum rank / min(n,m)")
        ax.set_ylim(0.0,.05)
    else:
        ax.set_yticks([50,100,150])
        ax.set_xlim(1e5,1e12)
        ax.set_ylabel("maximum rank")
    
    ax.tick_params(labeltop=False, labelright=True)
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.set_xticks([1e5,1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    ax.grid(which="major", axis="y")
    plt.tight_layout()
    plt.show()

#
#   Network Stats
#
def LVGNA_network_stats():
    med = np.median(vertex_counts.values())
    min = np.minimum(vertex_counts.values())
