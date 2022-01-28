from re import L
from matplotlib import lines
from matplotlib import patches
from plottingStyle import * 

T_color = t1_color
T_linestyle = "solid"
LT_color = t2_color
LT_linestyle = "--"
LT_rom_color = darkest_t2_color
LT_rom_linestyle=(0,(10,1.5))

LREigenAlign_color = t3_color
LREigenAlign_linestyle = ":"
LREA_Tabu_color = "k"
LREA_Tabu_linestyle = "solid"
LREA_Klau_color = "g"
LREA_Klau_linestyle= "solid"

LGRAAL_color = t5_color
LGRAAL_linestyle = (0,(3,1,1,1,1,1))
LRT_color = t4_color
LRT_linestyle = (0, (3, 1, 1, 1))
LRT_lrm_color = t6_color
LRT_lrm_linestyle = "-."

LT_Klau_color = t3_color
LT_Klau_linestyle = (0,(3,1,1,1,1,1))
LT_Tabu_color = t6_color
LT_Tabu_linestyle = (0,(10,1.5))

LRT_Klau_color = t1_color
LRT_Klau_linestyle = ":"
LRT_Tabu_color = t5_color
LRT_Tabu_linestyle = "solid"


checkboard_color = [.95]*3

def call_all_plots():
    #
    #  -- Low Rank Structure -- #
    #
    max_rank_experiments_v3()
    TAME_vs_LRTAME_clique_scaling_detailed()
        #TODO: need to update with 8 + 9 Clique results
        #      violin plots style need to be updated too. 

    # Appendix plots
    TAME_LVGNA_rank_1_case_singular_values()
    TAME_RandomGeometric_rank_1_case_singular_values()
        #TODO: need to be updated with the same noise models as max-rank exps


    #
    #  -- LVGNA Experiments --  #
    #
    make_LVGNA_runtime_performance_plots_v2()
    
    #
    #  -- Random Graph Experiments --  #
    #
    RandomGeometricRG_PostProcessing_is_needed()
    #Appendix Plots 
    RandomGeometricDupNoise_KlauPostProcessing()
    RandomGeometricERNoise_KlauPostProcessing()
        # TODO: add in LREA post processing results 
        #       update plot style 
        # NOTE: default params are n = 250, p = .05

    #
    #  -- Post Processing --  #
    #
    klauKNearestRedo()
        #TODO: data needs to be rerun with duplication noise sp:25
    LVGNA_PostProcessing_localizedData()
        #NOTE: LREA post processed results are difficult to read. 

#
#   TAME Low Rank Structure + Scaling
#

#  - LRTAME has more accurate singular values for Triangle Adjancency tensors -  #
def TAME_LVGNA_rank_1_case_singular_values(axes=None):
    with open(TAME_RESULTS + "Rank1SingularValues/TAME_LVGNA_iter_30_no_match_tol_1e-12.json","r") as f:
        TAME_data = json.load(f)
        TAME_data= TAME_data[-1]

    with open(TAME_RESULTS+ "Rank1SingularValues/LowRankTAME_LVGNA_iter_30_no_match_tol_1e-12.json","r") as f:
        LowRankTAME_data = json.load(f)    
        LowRankTAME_data = LowRankTAME_data[-1]


    if axes is None:
        f,axes = plt.subplots(1,1,dpi=60)
        f.set_size_inches(3, 3)

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


    ax = axes

    ax.set_yscale("log")
    ax.grid(which="major", axis="y")

    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=t3_color)
    ax.annotate("TAME", xy=(.2,.7),xycoords='axes fraction',fontsize=12,c=T_color)
    ax.annotate("LowRankTAME", xy=(.2,.1),xycoords='axes fraction',fontsize=12,c=LRT_color)
    ax.set_ylabel(r"$\sigma_2$")
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")

    
    ax.set_xlim(1e5,1e13)
    ax.set_ylim(1e-16,1e-7)

    #scatter plot formatting
    marker_size = 15
    marker_alpha = 1.0

    ax.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=T_color,s=marker_size)
    ax.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size)

    #plot machine epsilon
    ax.plot([1e5,1e13],[2e-16]*2,c=t3_color,zorder=1)

    ax.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=LRT_color,s=marker_size)
    scatter = ax.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=LRT_color,s=marker_size,zorder=2)
    
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
    #ax = plt.subplot(122)

    #format the axis
    ax.set_yscale("log")
    ax.grid(which="major", axis="y")
    #ax.set_ylabel(r"max $\sigma_2")
    #ax.set_ylabel(r"max [$\sum_{i=2}^k\sigma_i]")
    #ax.yaxis.set_ticks_position('right')
    #ax.tick_params(labeltop=False, labelright=True)
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(3e5,7e11)
    ax.set_xticks([1e7,1e8,1e11])
    #ax.set_ylim(1e-16,1e-7)
    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=t3_color)
    ax.set_ylabel(r"$\sigma_2$")
    #scatter plot formatting
    marker_size = 20
    marker_alpha = 1.0

    #plot the TAME Data
    """
    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=darker_t4_color,s=marker_size)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=darker_t4_color,s=marker_size)
    """
    #plot machine epsilon
    plt.plot([3e5,7e11],[2e-16]*2,c=t3_color,zorder=1)

    #plot LowRankTAME Data
    plt.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=LRT_color,s=marker_size,zorder=2)
    plt.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=LRT_color,s=marker_size,zorder=2)

    #plot TAME Data
    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=T_color,s=marker_size,zorder=3)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size,zorder=3)


    axins = ax.inset_axes([.6,.15,.25,.25]) # zoom = 6
    axins.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=T_color,s=marker_size)
    axins.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=T_color,s=marker_size)
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

   
    #axins.tick_params(labelleft=False, labelbottom=False)
    #axins.set_yticks([])

    """
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    """


    #TODO: combining plots, potentially remove later
    plt.tight_layout()
    plt.show()




#  - TAME iterates are low rank -  #

def max_rank_experiments_v3():

    #f = plt.figure(dpi=60)
    #f.set_size_inches(10, 4)
    f = plt.figure(figsize=(5.2,5))
    n = 3 
    m = 2 
    gs = f.add_gridspec(nrows=n, ncols=m, left=0.05, right=0.975,wspace=0.225,hspace=0.1,top=.975,bottom=.1)
    all_ax = np.empty((n,m),object)
    for i in range(n):
        for j in range(m):
            if j == 0:
                all_ax[i,j] = f.add_subplot(gs[i,j])
            else:
                all_ax[i,j] = f.add_subplot(gs[i,j],sharex=all_ax[i,0])


    DupNoise_data = "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.5]_noiseModel:Duplication_sp:[0.25]_trials:50_MaxRankResults.json"
    ERNoise_data = "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.05]_noiseModel:ER_trials:50_MaxRankResults.json"
    max_rank_synthetic_data_v2(all_ax[0,:],ERNoise_data,"ER")
    max_rank_synthetic_data_v2(all_ax[1,:],DupNoise_data,"Duplication")
    max_rank_LVGNA_data_v2(all_ax[2,:])
    
    checkboard_color = [.925]*3

    for ax in all_ax.reshape(-1):
        ax.grid(which="major", axis="both")
        ax.set_xscale("log")
        ax.set_xlim(1e5,1e12)

    #
    # -- handle tick marks -- #
    #
    for i,ax in enumerate(all_ax[:,1]):
        ax.set_ylabel("max rank\n min{n,m}",ha="center")
        #ax.set_ylabel(r"max rank/$\min{\{n,m\}}$")
        ax.annotate('', xy=(-.1, .25), xycoords='axes fraction', xytext=(-.1, 0.75),
                        arrowprops=dict(arrowstyle="-", color='k'))
        
        if i != 2:
            tick_ha = "right"
        else:
            tick_ha = "left"
        
        for tick in ax.yaxis.get_majorticklabels():
                tick.set_horizontalalignment(tick_ha)
        ax.tick_params(axis="y",which="both",direction="in",pad=-25)
        ax.set_yscale("log")
        if i != 2:
            ax.set_ylim(5e-3,1.5)
        else:
            ax.set_ylim(5e-4,2e-1)

        # -- add in Plot labels
        x_loc = .85
        title_size = 12
        if i == 0:
            ax.annotate(u"Erdős Rényi"+"\nNoise",xy=(.9,.675), xycoords='axes fraction',ha="right",fontsize=title_size)
        elif i == 1:
            ax.annotate("Duplication\n Noise",xy=(.9,.675), xycoords='axes fraction',ha="right",fontsize=title_size)
        elif i == 2:
            x_loc = .85
            ax.annotate("LVGNA",xy=(x_loc,.7), xycoords='axes fraction',ha="right",fontsize=title_size)

        
        #ax.yaxis.set_ticks_position('right')
        if i != 2:
            ax.tick_params(axis="x",direction="out",which='both', length=0)

    #ax.set_ylabel("maximum rank")
    all_ax[2,1].tick_params(labeltop=False, labelright=True)

    #  -- set labels --  #
    for i,ax in enumerate(all_ax[:,0]):
        ax.set_ylabel("max rank")
        ax.set_xlim(5e4,5e11)
        ax.set_xticks([1e5,1e6,1e7,1e8,1e9,1e10,1e11])
        ax.tick_params(axis="y",which="both",direction="in",pad=-25)
        if i != 2:
            ax.set_ylim(0,320)
            ax.tick_params(axis="x",direction="out",which='both', length=0)
            ax.set_yticks([50,100,150,200,250,300])
            ax.set_xticklabels([])
            #for tl in ax.get_yticklabels():
            #        tl.set_bbox(bbox)
        else:
            ax.set_ylim(0,150)
            ax.set_yticks([50,100,125])
            ax.set_xticklabels([1e5,1e6,1e7,1e8,1e9,1e10,1e11])
        
        # -- add in annotations  

    all_ax[2,0].xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())

    for j in range(m):
        #parity = 1
        for (i,ax) in enumerate(all_ax[:,j]):
            parity = 1
            for pos in ['left','bottom','top','right']:
                ax.spines[pos].set_visible(False)
            if j == 1:
                ax.tick_params(axis="y",which='minor', length=0)

            if i == 2:
                ax.yaxis.set_ticks_position('right')
            """
            if i == 0:
                ax.spines['bottom'].set_visible(False)
            elif i == 2:
                ax.spines['top'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
            else:
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
            """
            if parity == -1:
                all_ax[i,j].patch.set_facecolor(checkboard_color)
            
            if parity == 1:
                bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
            else:
                bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=.1)
   
            for tl in ax.get_yticklabels(): 
                tl.set_bbox(bbox)
            parity *= -1
        parity *= -1


            
    for ax in all_ax[2,:]:
        ax.set_xlabel(r"$|T_A||T_B|$")

    plt.show()

def max_rank_LVGNA_data_v2(axes):
    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_LVGNA.json", 'r') as f:
        MM_results  = json.load(f)
    MM_exp_data, param_indices,MM_file_names = process_TAME_output2(MM_results[1])

    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_Biogrid.json","r") as f:
        Biogrid_results = json.load(f)

    Biogrid_exp_data, _ = process_biogrid_results(Biogrid_results)
    label_meta = [
        ((.05, .35),"o",t1_color,"solid"),
        ((.05, .575),"v",t2_color,"dotted"),
        ((.075, .75),"*",t3_color,"dashed"),
        ((.475, .75),"s",t4_color,"dashdot")]


    for (param, j),(loc,marker,c,linestyle) in zip(param_indices.items(),label_meta):
        n_points = []
        max_ranks = []
        normalized_max_ranks = []
        param_label = f"β:{int(float(param.split('β:')[-1]))}"
        axes[0].annotate(param_label, xy=loc, xycoords='axes fraction',c=c)

        for i in range(MM_exp_data.shape[0]):

            graph_names = [str.join(" ", x.split(".ssten")[0].split("_")) for x in MM_file_names[i]]
            min_n = np.min([vertex_counts[f] for f in graph_names])
            tri_counts = triangle_counts[graph_names[0]]*triangle_counts[graph_names[1]]
           # tri_counts = sum([triangle_counts[graph_names[0]] for f in graph_names])
            n_points.append(tri_counts)
            max_ranks.append(np.max(MM_exp_data[i,:,j,:]))
            normalized_max_ranks.append(np.max(MM_exp_data[i,:,j,:])/min_n)

            
        n_points.append(407650*347079)
            #Biogrid_human_net: |V|=14867, |T| = 407650 
            #Biogrid_yeast_net: |V|=5850,  |T| = 347079
        max_ranks.append(np.max(Biogrid_exp_data[:,j,:]))
        normalized_max_ranks.append(np.max(Biogrid_exp_data[:,j,:])/5850)

        #ax.scatter(n_points,max_ranks,c=c,s=15,alpha=.4,marker=marker)
        plot_1d_loess_smoothing(n_points,max_ranks,.3,axes[0],c=c,linestyle=linestyle)
        plot_1d_loess_smoothing(n_points,normalized_max_ranks,.3,axes[1],c=c,linestyle=linestyle,logFilter=True)
       

    (filename,color,label,loc,linestyle) = (
        "LVGNAMaxEigShiftRanks_alphas:[0.5,1.0]_iter:30_SSHOPMSamples:1000_tol:1.0e-16_results.json",
        [.25]*3,
        "β:"+r"$\lambda_A\lambda_B$",
        (-.05,.02),
        (0,(3,1,1,1,1,1)))
    
    with open(TAME_RESULTS + "LVGNA_Experiments/"+filename,"r") as f:
        data = json.load(f)

        tri_products = []
        max_ranks = []
        normalized_max_ranks = []

        tri_count =lambda f: triangle_counts[str.join(" ", f.split(".ssten")[0].split("_"))]
        for (graphA, graphB,shiftA,shiftB,profiles) in data:

            graph_names = [str.join(" ", x.split(".ssten")[0].split("_")) for x in [graphA, graphB]]
            min_n = np.min([vertex_counts[f] for f in graph_names])
            
            tri_products.append(tri_count(graphA)*tri_count(graphB))

            max_ranks.append(max([max(profile["ranks"]) for (p,profile) in profiles]))
            normalized_max_ranks.append(max([max(profile["ranks"]) for (p,profile) in profiles])/min_n)



        #ax.annotate(r"$\beta =\lambda_A^*\lambda_B^*$", xy=(.8,.4), xycoords='axes fraction',c=color)
        #axes[0].scatter(tri_products,max_ranks,c=color,s=15,alpha=.5,marker="s")
        plot_1d_loess_smoothing(tri_products,max_ranks,.3,axes[0],c=color,linestyle=linestyle,logFilter=True)
        plot_1d_loess_smoothing(tri_products,normalized_max_ranks,.3,axes[1],c=color,linestyle=linestyle,logFilter=True)
        
        axes[0].annotate(label, xy=loc, xycoords='axes fraction',c=color)

def max_rank_synthetic_data_v2(axes,filename,noise_model="ER"):
    #with open(TAME_RESULTS + "MaxRankExperiments/","r") as f:
    with open(TAME_RESULTS + "MaxRankExperiments/" + filename,"r") as f:
    #with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_RandomGeometric_degreedist_log5_iter:15_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100,500,1K,2K,5K,10K,20K]_noMatching_pRemove:[.01,.05]_tol:1e-12_trialcount:50.json","r") as f:
        synth_results = json.load(f)
    
    MM_exp_data, n_vals, p_vals, param_vals, tri_counts = process_synthetic_TAME_output2(synth_results,noise_model)
 
    #ax = axes[0]
    #ax.set_xscale("log") 
    #ax.set_ylim(00,315)

    if filename == "LRTAME_noMatch_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0,10.0,100.0]_n:[100,500,1000,2000,5000,10000]_p:[0.05]_noiseModel:ER_trials:50_MaxRankResults.json":
        label_meta = [
            ((.825, .09),"o",t1_color,"solid"),
            ((.825, .26),"v",t2_color,"dotted"),
            ((.825, .6), "*",t3_color,"dashed"),
            ((.825, .8), "s",t4_color,"dashdot")]
    else:
        label_meta = [
            ((.85, .11), "o",t1_color,"solid"),
            ((.85, .25),"v",t2_color,"dotted"),
            ((.85, .55), "*",t3_color,"dashed"),
            ((.85, .825), "s",t4_color,"dashdot")]#  no_ylim_points

    #label_meta = [((.92, .25),t1_color),((.8, .51),t2_color),((.625, .8),t3_color),((.25, .88),t4_color)]

    #beta=100 annotations
    beta_100_meta = [(.15,.4),(.42,1.01),(.51,1.01),(.6,1.01),(.725,1.01),(.8,1.01),(.9,1.01)]

    #beta=10 annotations
    beta_10_meta = [(.15,.32),(.42,.515),(.51,.6),(.6,.65),(.725,.75),(.8,.95),(.9,.95)]

    for ((params,k),(loc,marker,c,linestyle)) in zip(param_vals.items(),label_meta):
        max_vals = []
        normalized_max_vals = []
        mean_tris = []
        
        for n,j in n_vals.items():
            max_vals.append(np.max(MM_exp_data[:,j,:,k,:]))
            normalized_max_vals.append(np.max(MM_exp_data[:,j,:,k,:])/n)
            mean_tris.append(np.mean(tri_counts[:,j,:,k]))
        

        axes[0].plot(mean_tris,max_vals,c=c,linestyle=linestyle)
        axes[1].plot(mean_tris,normalized_max_vals,c=c,linestyle=linestyle)

        param_label = f"β:{int(float(params.split('β:')[-1]))}"
        axes[0].annotate(param_label, xy=loc, xycoords='axes fraction',c=c)

#  - TAME & LRTAM don't scale for larger motifs -  #

def TAME_vs_LRTAME_clique_scaling_detailed():
    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}\usepackage{amsmath}')

    #
    #  subplot routines 
    #

    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8):
        v = ax.violinplot(data, points=100, positions=[.5], showmeans=False, 
                       showextrema=False, showmedians=True,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y0-.1),(x0,y0 + (y1-y0)/2 -.1)],[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.38),xycoords="axes fraction",ha="center",fontsize=10)
        ax.annotate(f"{np.min(data):.{precision}f}",xy=(0,.7),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
        ax.annotate(f"{np.max(data):.{precision}f}",xy=(1,.01),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)

        if c is not None:
            v["cmedians"].set_color("k")
            v["cmedians"].set_alpha(.3)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)            
                b.set_alpha(v_alpha)
                #b.set_alpha(1)
                b.set_color(c)

    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(10)
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.5], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y0-.1),(x0,y0 + (y1-y0)/2 -.1)],[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("med.",xy=(.5,.5),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(0,.7),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
        ax.annotate(f"max",xy=(1,.01),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.2)
                b.set_color(c)


    #
    #  Parse Data
    #

    results_path = TAME_RESULTS + "RandomGraphTAME_dupNoise/TAME_clique_scaling_exps/"

    elemwise_list_sum =lambda l1,l2: [a + b for (a,b) in zip(l1,l2)]
    #filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[25]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json" 
    filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7,8]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"

    with open(results_path + filename,"r") as file:
        data = json.load(file)
        LR_results = {}
        debug_results = {}
        LR_seeds = {}

        for (order, LRTAME_output) in data:
            trials = len(LRTAME_output)
            LR_results[order] = {
                "full runtime":{},
                "contraction runtime":{},
                "A motifs":[],
                "ranks":{},
            }
            debug_results[order] = {
                "runtimes":[],
                "ranks":[]
            }
            LR_seeds[order] = []
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(LRTAME_output):
                LR_results[order]["A motifs"].append(A_motifs)
                LR_seeds[order].append(seed)
                for (params,profile) in profiling:

                    rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])
                    if params in LR_results[order]["full runtime"]:
                        LR_results[order]["full runtime"][params].append(rt)
                    else:
                        LR_results[order]["full runtime"][params] = [rt]

                    contract_rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        ])
                    if params in LR_results[order]["contraction runtime"]:
                        LR_results[order]["contraction runtime"][params].append(contract_rt)
                    else:
                        LR_results[order]["contraction runtime"][params] = [contract_rt]


                    if params == 'α:0.5_β:1.0':
                        debug_results[order]["ranks"].append(profile["ranks"])
                        debug_results[order]["runtimes"].append(contract_rt)

                    if params in LR_results[order]["ranks"]:
                        LR_results[order]["ranks"][params].append(profile["ranks"])
                    else:
                        LR_results[order]["ranks"][params] = [profile["ranks"]]

            # average the runtime results 
            #for (param,rts) in LR_results[order]["contraction runtime"].items():
            #    LR_results[order]["contraction runtime"][param] = [rt/trials for rt in LR_results[order]["contraction runtime"][param]]
            
            for key in ["contraction runtime","full runtime"]:
                for (param,rts) in LR_results[order][key].items():
                    LR_results[order][key][param] = np.median(np.array(rts),axis=0)
                    
                #LR_results[order]["full runtime"][param] = [rt/trials for rt in LR_results[order]["full runtime"][param]]
    #return debug_results

    file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[25]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)

        T_results = {}


        for (order, TAME_output) in data:
            trials = len(TAME_output)
            T_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
 
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(TAME_output):
                for (params,profile) in profiling:
                    rt = reduce(elemwise_list_sum,[
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])

                    if param in T_results[order]:
                        T_results[order]["full runtime"][params].append(rt)
                    else:
                        T_results[order]["full runtime"][params] = [rt]

                    if param in T_results[order]:
                        T_results[order]["contraction runtime"][params].append(profile["contraction_timings"])
                    else:
                        T_results[order]["contraction runtime"][params] = [profile["contraction_timings"]]

            for key in ["contraction runtime","full runtime"]:
                for (param,rts) in T_results[order][key].items():
                    T_results[order][key][param] = np.median(np.array(rts),axis=0)
            """
            for params in T_results[order]["contraction runtime"].keys():
                T_results[order]["contraction runtime"][params] = [rt/trials for rt in T_results[order]["contraction runtime"][params]]
            for params in T_results[order]["full runtime"].keys():
                T_results[order]["full runtime"][params] = [rt/trials for rt in T_results[order]["full runtime"][params]]
            """
    
    file = "LambdaTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)
        LT_results = {}

        all_shifts = [
            'α:0.5_β:0.0',
            'α:0.5_β:1.0',
            'α:1.0_β:0.0',
            'α:1.0_β:1.0',
        ]
        for (order, TAME_output) in data:
            trials = len(TAME_output)
            LT_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
            print(f"trials: {trials}")
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profile) in enumerate(TAME_output):

                #  aggregate the runtimes
                full_rt = reduce(elemwise_list_sum,[
                        profile["Matching Timings"],
                        profile["TAME_timings"],
                        profile["Scoring Timings"],
                        ])

                for (i,rt) in enumerate(full_rt):
                    params = all_shifts[i]
                    if params in LT_results[order]["full runtime"]:
                        LT_results[order]["full runtime"][params].append(rt)
                    else:
                        LT_results[order]["full runtime"][params] = [rt]

                for (i,rt) in enumerate(profile["TAME_timings"]):
                    params = all_shifts[i]
                    if params in LT_results[order]["contraction runtime"]:
                        LT_results[order]["contraction runtime"][params].append(rt)
                    else:
                        LT_results[order]["contraction runtime"][params] = [rt]
        
            for key in ["contraction runtime","full runtime"]:
                for (param,rts) in LT_results[order][key].items():
                    LT_results[order][key][param] = np.median(np.array(rts))


    #return LR_results, T_results

    #
    #  Create Plots 
    #
    fig = plt.figure(figsize=(5.2,3))
    global_ax = plt.gca()
    global_ax.set_axis_off()


    parity = 1  #used for checkerboard effect in plots.             
    

    linestyles= {
        'α:0.5_β:0.0':"dotted",
        'α:0.5_β:1.0':"solid",
        'α:1.0_β:0.0':(0,(3,1,1,1)),
        'α:1.0_β:1.0':(0,(5,1))
    }

    motif_label = 0
    rt_data = 1
    rank_data = 2
    A_motif_data = 3

    n = 4
    m = len(T_results.keys())

    widths = [.5,3,2,1]
    col_width_ratios = [1]*m
    col_width_ratios.append(.8) 

                                # +1 for vioin legend
    gs = fig.add_gridspec(nrows=n,ncols=m + 1,hspace=0.1,wspace=0.1,
                          height_ratios=widths,width_ratios=col_width_ratios,
                          left=.18,right=.99,top=.95,bottom=.05)
                                        
    gs_ax = np.empty((n,m),object)
    iterate_tick_idx = 2


    # -- plot runtime data -- # 
    for idx,((LRT_order,LRT_data),(T_order,T_data),(LT_order,LT_data)) in enumerate(zip(LR_results.items(),T_results.items(),LT_results.items())):
        assert LRT_order == T_order
        assert LRT_order == LT_order

        order = LRT_order
        for row in range(n):
            ax = fig.add_subplot(gs[row,idx])
            gs_ax[row,idx] = ax


            if row == motif_label:
                ax.annotate(f"{order}",xy=(.5, .5), xycoords='axes fraction', c="k",size=10,ha="center",va="center")

            elif row == rt_data: 
                for (exp_data,c) in zip([LRT_data,T_data],[LRT_color,T_color]):
                    for (param,runtime) in exp_data["contraction runtime"].items():
                        if param == 'α:0.5_β:0.0' or param == 'α:1.0_β:1.0':
                            print("skipping")
                            continue
                        gs_ax[row,idx].plot(runtime,c=c,linestyle=linestyles[param])
    
                print(len(runtime))
                iterations = len(runtime)
                for (param,runtime) in LT_data["contraction runtime"].items():
                    if param == 'α:0.5_β:0.0' or param == 'α:1.0_β:1.0':
                            continue
                    print(runtime)
                    gs_ax[row,idx].axhline(runtime/iterations,c=LT_color,linestyle=linestyles[param])
                
                #for (param,runtime) in T_data["contraction runtime"].items():
                #    if param == 'α:0.5_β:0.0' or param == 'α:1.0_β:1.0':
                #        continue
                #    ax.plot(runtime,c=T_color,linestyle=linestyles[param])

            elif row == rank_data:
                for (param,ranks) in LRT_data["ranks"].items():
                    if param == 'α:0.5_β:0.0' or param == 'α:1.0_β:1.0':
                        continue
                    if idx == iterate_tick_idx:
                        ax.set_zorder(3.5)
                    ax.plot(np.median(np.array(ranks),axis=0),c=[.1]*3,linestyle=linestyles[param])
                    
            elif row == A_motif_data:
                if idx % 2 == parity:
                    make_violin_plot(ax,LRT_data["A motifs"],precision=0,c="w")
                    ax.patch.set_facecolor('k')
                    ax.patch.set_alpha(0.1)
                else:
                    make_violin_plot(ax,LRT_data["A motifs"],precision=0,c="k",v_alpha=.1)



    # -- plot motif data -- # 

    #
    #  Adjust Axes
    #
    gs_ax[motif_label,0].set_ylabel(" Clique Size",rotation=0,labelpad=32.5,ha="center",va="center")
    
    subylabel_xpos = -.55 
    gs_ax[rt_data,0].set_ylabel("Contraction\nRuntime (s)",rotation=0,labelpad=32.5,ha="center",va="center")
    gs_ax[rt_data,0].annotate("25 unique\ntrials",xy=(subylabel_xpos, .15), xycoords='axes fraction',ha="center",fontsize=7,style='italic')
    
    #gs_ax[0,0].annotate(r"$25\,unique$"+"\n"+r"$trials$",xy=(-.5, .175), xycoords='axes fraction',ha="center",fontsize=8)
    for (idx,ax) in enumerate(gs_ax[rt_data,:]):
        ax.set_ylim(5e-5,5e3)
        ax.set_yscale("log")
        
        ax.yaxis.set_ticks_position('right')
        ax.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])

        if idx == m-1:
            ax.set_yticklabels([r"$10^{-4}$",r"$10^{-3}$",None,r"$10^{-1}$",None,r"$10^1$",None,r"$10^3$"])
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,14])
        ax.grid(True)
        ax.set_xlim(-1,15)

    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
    #bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=-.1)
    gs_ax[rt_data,1].annotate("LR-TAME",c=LRT_color,xy=(.225, .375), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[rt_data,1].annotate("TAME",c=T_color,xy=(.2, .725), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[rt_data,1].annotate(r"$\Lambda$"+"-TAME",c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)

    gs_ax[rank_data,0].set_ylabel("Iterate\nRank",rotation=0,labelpad=32.5,ha="center",va="center")
    gs_ax[rank_data,0].annotate("max rank=100",xy=(subylabel_xpos, .15), xycoords='axes fraction',ha="center",fontsize=7,style='italic')
    
    for (idx,ax) in enumerate(gs_ax[rank_data,:].reshape(-1)):
        ax.set_ylim(-1,32.5)
        ax.yaxis.set_ticks_position('right')
        ax.set_yticks([0,1,5,10,15,20,25,30])
        if idx == m-1:
            ax.set_yticklabels([None,None,5,None,15,None,25,None])
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,14])
        ax.grid(True)
        ax.set_xlim(-1,15)

    # -- add in shift annotations -- #

    #gs_ax[rank_data,2].annotate('α=.5  β=0',xy=(.225, .25), xycoords='axes fraction',fontsize=6)

    """
    # NOTE:old version designed for > 2 labels
    algoLabels = [
        ('α=.5 β=1\n(both shifts)',None,(.975, .65),),
        ('α=1 β=0 (no shifts)',None,(.975, .05)),
    ]
    for (label,sublabel,xy_loc) in algoLabels:
        #bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=.1)
        if sublabel is None:
            gs_ax[rank_data,1].annotate(label,xy=xy_loc, xycoords='axes fraction',fontsize=6,ha="right")
        else:
            gs_ax[rank_data,1].annotate(label,xy=xy_loc, xycoords='axes fraction',fontsize=6,ha="right")
    """
    gs_ax[rank_data,1].annotate('α=.5 β=1\n(both shifts)',xy=(.375, .65), xycoords='axes fraction',fontsize=6,ha="left")
    
    gs_ax[rank_data,1].annotate('α=1',xy=(.275, .16), xycoords='axes fraction',fontsize=6,ha="left")
    gs_ax[rank_data,1].annotate('β=0 (no shifts)',xy=(.275, .06), xycoords='axes fraction',fontsize=6,ha="left")

 



 
    legend_ax = fig.add_subplot(gs[A_motif_data,-1])
    make_violin_plot_legend(legend_ax)

    gs_ax[A_motif_data,0].set_ylabel("A motifs",rotation=0,labelpad=32.5,ha="center",va="center")
    gs_ax[A_motif_data,0].annotate("samples="+r"$10^4$",xy=(subylabel_xpos, .01), xycoords='axes fraction',ha="center",fontsize=7,style='italic')
    
    for ax in chain(gs_ax[[A_motif_data,motif_label],:].reshape(-1),[legend_ax]):
        ax.set_yticklabels([])

    for ax in chain(gs_ax.reshape(-1),[legend_ax]):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_xticklabels([])

    #add back in right most column tick marks
    gs_ax[rt_data,m-1].tick_params(axis="y",direction="out",which='both', length=2)
    gs_ax[rank_data,m-1].tick_params(axis="y",direction="out",which='both', length=2)

    #gs_ax[rank_data,-2].set_yticklabels([1,5,10,25])

    #gs_ax[rank_data,iterate_tick_idx].set_zorder(10)
    gs_ax[rank_data,iterate_tick_idx].xaxis.set_label_position("top")
    gs_ax[rank_data,iterate_tick_idx].xaxis.set_ticks_position('top')
    gs_ax[rank_data,iterate_tick_idx].tick_params(axis="x",direction="out", pad=-17.5,length=5)
    gs_ax[rank_data,iterate_tick_idx].set_xticklabels([1,5,15])
    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
    gs_ax[rank_data,iterate_tick_idx].annotate("iteration "+r"$(\ell)$",xy=(.5,1.25),ha="center",xycoords='axes fraction',fontsize=10)#.set_bbox(bbox)
    #gs_ax[rank_data,iterate_tick_idx].set_yticklabels([])
                                    # BUG(?) This needed to be called again 

    """
    parity = 1
    for i in range(n-1): #don't do the violin plot rows
        for j in range(m):  
            if parity == 1:
                gs_ax[i,j].patch.set_facecolor(checkboard_color)
                #all_gs_ax[i,j].patch.set_alpha(1.1)
            parity *= -1
        if m % 2 == 0:
            parity *= -1
    """

    plt.show()

def TAME_vs_LRTAME_clique_scaling_summarized():
    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}\usepackage{amsmath}')

    #
    #  subplot routines 
    #

    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8,format="default",xlim=None,xscale="linear"):

       
        #background_v = ax.violinplot(data, points=100, positions=[0.5], showmeans=False, 
        #                showextrema=False, showmedians=False,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #
        #positions=[0.5], ,widths=.5
        if xscale=="linear":
            v = ax.violinplot(data, points=100, showmeans=False,
                        showextrema=True, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        elif xscale=="log":
            v = ax.violinplot(np.log10(data), points=100, showmeans=False,
                    showextrema=True, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")
        ax.set_ylim(0.95, 1.3)
        #ax.set_xlim(np.min(data),np.max(data))

    

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0 + (y1-y0)/2+.1),(x0,y1+0.01)]]# [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]
        newMedianLines = [[(x0,y1+.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- place extremal markers underneath
        v['cbars'].set_segments([]) # turns off x-axis spine
        for segment in [v["cmaxes"],v["cmins"]]:
            ((x,y0),(_,y1)) = segment.get_segments()[0]
            segment.set_segments([[(x,0.45),(x,.525)]])
            segment.set_color(c)

        # -- write data values as text
        extremal_tick_ypos = -.175
        if format == "default":
            ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        elif format == "scientific":
            ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        else:
            print(f"expecting format to be either 'default' or 'scientific', got:{format}")


        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor(c)
                b.set_edgecolor(c)            
                b.set_alpha(v_alpha)
                b.set_color(c)

                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .04
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])
                clip_to_top_of_violin(v["cmaxes"])
                clip_to_top_of_violin(v["cmins"])
 
    def make_violin_plot_legend(ax,c="k"):
        
      

        np.random.seed(12)#Ok looking:12
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.5], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        ax.set_ylim(.5,1.0)
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        print(newMedianLines )
        v["cmedians"].set_segments(newMedianLines)
        #v['cbars'].set_segments([]) 
        #for segment in [v["cmaxes"],v["cmins"]]:
        #    ((x,y0),(_,y1)) = segment.get_segments()[0]
        #    segment.set_segments([[(x,0.45),(x,.525)]])
        #    segment.set_color(c)


        ax.annotate("median",xy=(.5,.2),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(.025,-.2),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,-.2),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.3)
                b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])
                #clip_to_top_of_violin(v["cmaxes"])
                #clip_to_top_of_violin(v["cmins"])

                """
                #  -- clip the top of the med-line to the top of the violin plot --  #
                med_line_x = v["cmedians"].get_paths()[0].vertices[0, 0]

                # find the y-vals of the violin plots near the med-lines
                distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                k = 5
                closest_x_points = np.argpartition(distances,k)[:k]
                new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )

                #clip the lines 
                v["cmedians"].get_paths()[0].vertices[:, 1] = np.clip(v["cmedians"].get_paths()[0].vertices[:, 1],-np.inf,new_max_y)
                """

    #
    #  Parse Data
    #

    results_path = TAME_RESULTS + "RandomGraphTAME_dupNoise/TAME_clique_scaling_exps/"

    elemwise_list_sum =lambda l1,l2: [a + b for (a,b) in zip(l1,l2)]
    #filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[25]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json" 
    filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7,8]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"

    with open(results_path + filename,"r") as file:
        data = json.load(file)
        LR_results = {}
        debug_results = {}
        LR_seeds = {}

        for (order, LRTAME_output) in data:
            trials = len(LRTAME_output)
            LR_results[order] = {
                "full runtime":{},
                "contraction runtime":{},
                "A motifs":[],
                "ranks":{},
            }
            debug_results[order] = {
                "runtimes":[],
                "ranks":[]
            }
            LR_seeds[order] = []
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(LRTAME_output):
                LR_results[order]["A motifs"].append(A_motifs)
                LR_seeds[order].append(seed)
                for (params,profile) in profiling:
                    """
                    rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])
                    if params in LR_results[order]["full runtime"]:
                        LR_results[order]["full runtime"][params].append(rt)
                    else:
                        LR_results[order]["full runtime"][params] = [rt]
                    """
                    contract_rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        ])
                    if params in LR_results[order]["contraction runtime"]:
                        LR_results[order]["contraction runtime"][params].append(np.max(contract_rt))
                    else:
                        LR_results[order]["contraction runtime"][params] = [np.max(contract_rt)]


                    if params == 'α:0.5_β:1.0':
                        debug_results[order]["ranks"].append(profile["ranks"])
                        debug_results[order]["runtimes"].append(contract_rt)

                    ranks = profile["ranks"]
                    if len(ranks) < 15:
                        # if algorithm terminated from tol bounds, extend last rank to fill the rest
                        ranks.extend([ranks[-1]]*(15-len(ranks)))

                    if params in LR_results[order]["ranks"]:
                        LR_results[order]["ranks"][params].append(profile["ranks"])
                    else:
                        LR_results[order]["ranks"][params] = [profile["ranks"]]

    #return debug_results

    file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[25]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)

        T_results = {}


        for (order, TAME_output) in data:
            trials = len(TAME_output)
            T_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
 
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(TAME_output):
                for (params,profile) in profiling:
                    """
                    rt = reduce(elemwise_list_sum,[
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])

                    if params in T_results[order]:
                        T_results[order]["full runtime"][params].append(np.sum(rt))
                    else:
                        T_results[order]["full runtime"][params] = [np.sum(rt)]
                    """
                    if params in T_results[order]["contraction runtime"]:
                        T_results[order]["contraction runtime"][params].append(np.max(profile["contraction_timings"]))
                    else:
                        T_results[order]["contraction runtime"][params] = [np.max(profile["contraction_timings"])]


    file = "LambdaTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)
        LT_results = {}

        all_shifts = [
            'α:0.5_β:0.0',
            'α:0.5_β:1.0',
            'α:1.0_β:0.0',
            'α:1.0_β:1.0',
        ]
        for (order, TAME_output) in data:
            trials = len(TAME_output)
            LT_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profile) in enumerate(TAME_output):

                #  aggregate the runtimes
                full_rt = reduce(elemwise_list_sum,[
                        profile["Matching Timings"],
                        profile["TAME_timings"],
                        profile["Scoring Timings"],
                        ])

                for (i,rt) in enumerate(full_rt):
                    params = all_shifts[i]
                    if params in LT_results[order]["full runtime"]:
                        LT_results[order]["full runtime"][params].append(rt)
                    else:
                        LT_results[order]["full runtime"][params] = [rt]

                for (i,rt) in enumerate(profile["TAME_timings"]):
                    params = all_shifts[i]
                    if params in LT_results[order]["contraction runtime"]:
                        LT_results[order]["contraction runtime"][params].append(rt)
                    else:
                        LT_results[order]["contraction runtime"][params] = [rt]

    #
    #  Create Plots 
    #
    fig = plt.figure(figsize=(7,3))
    global_ax = plt.gca()
    global_ax.set_axis_off()


    parity = 1  #used for checkerboard effect in plots.             
    

    linestyles= {
        'α:0.5_β:0.0':"dotted",
        'α:0.5_β:1.0':"solid",
        'α:1.0_β:0.0':(0,(3,1,1,1)),
        'α:1.0_β:1.0':(0,(5,1))
    }

    # column assignment

    motif_label = 0
    A_motif_data = 1
    rank_data = 2
    TAME_rt = 3
    LRTAME_rt = 4
    LTAME_rt = 5

    #rt_data = 1



    n = len(T_results.keys())
    m = 5 # - 1 for no LT 
    
    heights = [.4,.8,.8,1,1]
    row_height_ratios = [1]*n 
    #col_width_ratios.append(.8) 

    gs = fig.add_gridspec(nrows=n,ncols=m,hspace=.9,wspace=0.0, 
                          width_ratios=heights,height_ratios=row_height_ratios,
                          left=0.025,right=.975,top=.75,bottom=.025)
                                        
    gs_ax = np.empty((n,m),object)
    iterate_tick_idx = 2


    # -- plot runtime data -- # 
    motif_label_xloc = .4
    for idx,((LRT_order,LRT_data),(T_order,T_data),(LT_order,LT_data)) in enumerate(zip(LR_results.items(),T_results.items(),LT_results.items())):
        assert LRT_order == T_order
        assert LRT_order == LT_order

        order = LRT_order
        for col in range(m):

            """
            ax = fig.add_subplot(gs[idx,col])
            """
            if idx == 0:
                ax = fig.add_subplot(gs[idx,col])
            else:
                ax = fig.add_subplot(gs[idx,col],sharex=gs_ax[0,col])
            
            gs_ax[idx,col] = ax


            if col == motif_label:
                ax.annotate(f"{order}",xy=(motif_label_xloc, .375), xycoords='axes fraction', c="k",size=10,ha="center",va="center")#,weight="bold")
            elif col == TAME_rt:
                #ax.set_xscale("log")
                make_violin_plot(ax,T_data["contraction runtime"]['α:0.5_β:1.0'],precision=2,c=T_color,v_alpha=.3,xscale="log")
            elif col == LRTAME_rt:
                #ax.set_xscale("log")
                make_violin_plot(ax,LRT_data["contraction runtime"]['α:0.5_β:1.0'],precision=2,c=LRT_color,v_alpha=.3,xscale="log")

            elif col == LTAME_rt:
                make_violin_plot(ax,LT_data["contraction runtime"]['α:0.5_β:1.0'],
                                 precision=1,c=LT_color,v_alpha=.3,format = "scientific",xscale="log")
            elif col == A_motif_data:
                make_violin_plot(ax,LRT_data["A motifs"],precision=0,c="k",v_alpha=.2)

            elif col == rank_data:
                make_violin_plot(ax,np.array(LRT_data["ranks"]['α:0.5_β:1.0']).max(axis=1),precision=0,c="k",v_alpha=.2)
                #ax.set_xlim(13,61)

    # -- plot motif data -- # 

    #
    #  Adjust Axes
    #
    
    # -- Set the column titles -- #
    title_ypos = 1.4 
    annotation_ypos = .95
    gs_ax[0,motif_label].annotate("Clique Size",ha="center",va="bottom",xy=(motif_label_xloc+.1, title_ypos), xycoords='axes fraction')
    
    #subylabel_xpos = -.55 


    #bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=-.1)
    label_font = 11
    #gs_ax[0,LRTAME_rt].set_ylabel("Max Iterate\nTTV Time (s)",rotation=0,labelpad=30,ha="center",va="center")#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[0,LRTAME_rt].annotate("Longest\n Contraction (s)",rotation=0,ha="center",va="bottom",xy=(.5, title_ypos), xycoords='axes fraction')#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[1,LRTAME_rt].annotate("LR-TAME",rotation=0,ha="right",va="center",xy=(.9, annotation_ypos), xycoords='axes fraction',c=LRT_color,fontsize=label_font)#,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
    
    #gs_ax[LRTAME_rt,-1].set_ylabel("LR-TAME",rotation=0,labelpad=pad,ha="center",va="center")#,c=LRT_color,xy=(.225, .375), xycoords='axes fraction').set_bbox(bbox)
    #gs_ax[LRTAME_rt,-1].yaxis.set_label_position("right")

    gs_ax[0,TAME_rt].annotate("Longest\nContraction (s)",rotation=0,ha="center",va="bottom",xy=(.5, title_ypos), xycoords='axes fraction')#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[1,TAME_rt].annotate("TAME",rotation=0,ha="right",va="center",xy=(.9, annotation_ypos), xycoords='axes fraction',c=T_color,fontsize=label_font)
    #gs_ax[TAME_rt,-1].set_ylabel("TAME",rotation=0,labelpad=pad,ha="center",va="center")#,c=T_color,xy=(.2, .725), xycoords='axes fraction').set_bbox(bbox)
    #gs_ax[TAME_rt,-1].yaxis.set_label_position("right")

    #gs_ax[LTAME_rt,-1].set_ylabel(r"$\Lambda$"+"-TAME",rotation=0,labelpad=pad,ha="center",va="center")#,c=T_color,xy=(.2, .725), xycoords='axes fraction').set_bbox(bbox)
    #gs_ax[LTAME_rt,-1].yaxis.set_label_position("right")

    #gs_ax[LTAME_rt,0].set_ylabel("ttv time (s)",rotation=0,labelpad=30,ha="center",va="top")#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
    #gs_ax[0,LTAME_rt].annotate("Total",xy=(0.265, title_ypos), xycoords='axes fraction',ha="center",weight="bold")
    #gs_ax[0,LTAME_rt].annotate("Total Contraction\nTime (s)",xy=(0.5, title_ypos), xycoords='axes fraction',ha="center",va="bottom")
    #gs_ax[0,LTAME_rt].annotate("Time (s)",xy=(.5, title_ypos-.4), xycoords='axes fraction',ha="center",va="bottom")

    #gs_ax[1,LTAME_rt].annotate(r"$\Lambda$"+"-TAME",rotation=0,ha="left",va="center",xy=(0.0, annotation_ypos), xycoords='axes fraction',c=LT_color,fontsize=label_font)
    #gs
    gs_ax[0,rank_data].annotate("Max (LR-)TAME\nIterate Rank",ha="center",va="bottom",xy=(.5, title_ypos), xycoords='axes fraction')
    #gs_ax[rank_data,-1].annotate("max possible\nrank=100",xy=(1.4, .5), xycoords='axes fraction',ha="center",va="center",fontsize=7,style='italic')
    
    gs_ax[0,A_motif_data].annotate("A motifs",xy=(.5, title_ypos), xycoords='axes fraction',ha="center",va="bottom")

    # -- make a label for shared x-axis
    super_title_ypos = .875
    #annotation_ax = fig.add_axes([.075,super_title_ypos,.275,.075])
    #annotation_ax.patch.set_facecolor(t5_color)
    #annotation_ax.annotate("Real Axis in each Column",xy=(.5, .5), xycoords='axes fraction',ha="center",va="center",color="k")
    #annotation_ax.annotate(r"$|V_A|=100$           25 trials",xy=(1.05, 0.1), xycoords='axes fraction',ha="left",va="bottom",fontsize=label_font)
    

    legend_ax = fig.add_axes([.05,super_title_ypos,.15,.175],zorder=10)
    make_violin_plot_legend(legend_ax)  
    #legend_ax.annotate(r",xy=(3.0, 0.0), xycoords='axes fraction',ha="center",va="bottom",fontsize=label_font)
    
       
    #legend_ax = fig.add_axes([.4,title_ypos-.1,.6,title_ypos+.1],zorder=10)
    #legend_ax.patch.set_facecolor(checkboard_color)
    
    """
    gs_ax[A_motif_data,0].set_ylabel("A motifs",rotation=0,labelpad=32.5,ha="center",va="center")
    gs_ax[A_motif_data,0].annotate("samples="+r"$10^4$",xy=(subylabel_xpos, .01), xycoords='axes fraction',ha="center",fontsize=7,style='italic')
    """

    #for ax in gs_ax[:,TAME_rt]:
        #ax.set_xscale("log")
        #ax.set_xlim(1e-2)

    additional_ax = [
        global_ax, legend_ax # annotation_ax,
    ]

    for ax in chain(gs_ax.reshape(-1),additional_ax):
        ax.set_yticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_xticklabels([])
    """

   
    #add back in right most column tick marks
    gs_ax[rt_data,m-1].tick_params(axis="y",direction="out",which='both', length=2)
    gs_ax[rank_data,m-1].tick_params(axis="y",direction="out",which='both', length=2)

    #gs_ax[rank_data,-2].set_yticklabels([1,5,10,25])

    #gs_ax[rank_data,iterate_tick_idx].set_zorder(10)
    gs_ax[rank_data,iterate_tick_idx].xaxis.set_label_position("top")
    gs_ax[rank_data,iterate_tick_idx].xaxis.set_ticks_position('top')
    gs_ax[rank_data,iterate_tick_idx].tick_params(axis="x",direction="out", pad=-17.5,length=5)
    gs_ax[rank_data,iterate_tick_idx].set_xticklabels([1,5,15])
    bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
    gs_ax[rank_data,iterate_tick_idx].annotate("iteration "+r"$(\ell)$",xy=(.5,1.25),ha="center",xycoords='axes fraction',fontsize=10).set_bbox(bbox)
    #gs_ax[rank_data,iterate_tick_idx].set_yticklabels([])
                                    # BUG(?) This needed to be called again 

    """
    #for j in range(m):
    #    gs_ax[0,j].patch.set_facecolor(checkboard_color)

    #global_ax.add_patch(Rectangle((-3,0), 3, 4, facecolor=checkboard_color))
    plt.show()

def TAME_vs_LRTAME_clique_scaling_summarized_v2():
    """This version stacks the results of LR-TAME and TAME on top of one another."""

    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}\usepackage{amsmath}')

    #
    #  subplot routines 
    #
    extremal_tick_ypos = .125
        # subroutine globals
    def underline_text(ax,text,c,linestyle):
        tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color=c,linestyle=linestyle,linewidth=1.5,alpha=.8))

    def mark_as_algorithm(ax,text,c,linestyle,algorithm="LRTAME"):
        
        tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        
        #ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color=LRT_color,linestyle=LRT_linestyle,linewidth=1.5,alpha=.8))


        # calculate asymmetry of x and y axes:
        x0, y0 = fig.transFigure.transform((0, 0)) # lower left in pixels
        x1, y1 = fig.transFigure.transform((1, 1)) # upper right in pixes
        dx = x1 - x0
        dy = y1 - y0
        maxd = max(dx, dy)

        if algorithm == "LRTAME":
            radius=.02
            height = radius * maxd / dy
            width = radius * maxd / dx

            p=ax.add_patch(patches.Ellipse((tb.xmin-.015,tb.y0+(5/8)*tb.height),width, height,color=LRT_color,transform=fig.transFigure))
            
        elif algorithm == "TAME":
            side_length=.015
            height = side_length * maxd / dy
            width = side_length * maxd / dx

            p = ax.add_patch(patches.Rectangle((tb.xmax+.01,tb.y0+.5*(tb.height - side_length)),
                                           width, height,color=T_color,
                                           transform=fig.transFigure))
        else:
            raise ValueError(f"algorithm must be either 'TAME' or 'LRTAME', got {algorithm}.\n")
        #ax.add_patch(patches.Ellipse((tb.xmin-.015,tb.y0+tb.height/2),width, height,color=LRT_color,transform=fig.transFigure))
        #ax.add_patch(patches.Ellipse((tb.xmax,tb.y0),width, height,color=LRT_color,#transform=fig.transFigure),)
        """
        #width = .05
        #height = .01
        xshift = tb.width*.2
        height = .03
        width = .0075
        ax.add_patch(patches.Rectangle((tb.xmin-xshift,tb.y0 + (3/4)*tb.height),width, height,color=LRT_color,transform=fig.transFigure))

        height = .015 #* maxd / dy
        width = .02 #* maxd / dx
        xshift = tb.width*.1
        ax.add_patch(patches.Rectangle((tb.xmin-xshift,tb.y1), width,height,color=LRT_color,transform=fig.transFigure))
        """
    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8,format="default",xlim=None,xscale="linear",column_type=None):

       
        #background_v = ax.violinplot(data, points=100, positions=[0.5], showmeans=False, 
        #                showextrema=False, showmedians=False,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #
        #positions=[0.5], ,widths=.5
        if xscale=="linear":
            v = ax.violinplot(data,[.5], points=100, showmeans=False,widths=.15,
                        showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        elif xscale=="log":
            v = ax.violinplot(np.log10(data), points=100, showmeans=False,widths=.15,showextrema=False, showmedians=True,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        else:
            raise ValueError(f"supports xscale values: 'linear' & 'log'; Got {xscale}.")
        #ax.set_ylim(0.95, 1.3)
        #ax.set_xlim(np.min(data),np.max(data))

    

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0 + (y1-y0)/2+.1),(x0,y1+0.01)]]# [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]
        print(f"y0:{y0},y1:{y1}")
        newMedianLines = [[(x0,y1),(x0,y1+.7)]]
        v["cmedians"].set_segments(newMedianLines)

        # -- place extremal markers underneath
        """
        v['cbars'].set_segments([]) # turns off x-axis spine
        for segment in [v["cmaxes"],v["cmins"]]:
            ((x,y0),(_,y1)) = segment.get_segments()[0]
            segment.set_segments([[(x,0.45),(x,.525)]])
            segment.set_color(c)
        """

        # -- write data values as text
        
        if column_type is None:

            if format == "default":
                ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10)
                ax.annotate(f"{np.min(data):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
                ax.annotate(f"{np.max(data):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            elif format == "scientific":
                ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.4),xycoords="axes fraction",ha="center",fontsize=10)
                ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
                ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            else:
                print(f"expecting format to be either 'default' or 'scientific', got:{format}")
        elif column_type == "merged_axis":
            pass
        else:
            raise ValueError("column_type expecting 'merged_axis' or None, but got {column_type}\n")

        if c is not None:
            v["cmedians"].set_color(c)
            v["cmedians"].set_alpha(.75)
            for b in v['bodies']:
                # set colors 
                b.set_facecolor("None")
                b.set_edgecolor(c)            
                b.set_alpha(v_alpha)
                #b.set_color(c)

                #  -- only plot the top half of violin plot --  #
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)
                
                #  -- clip the top of the med-line to the top of the violin plot --  #
                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    #new_max_y += .04
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])
                #clip_to_top_of_violin(v["cmaxes"])
                #clip_to_top_of_violin(v["cmins"])
 
    def make_violin_plot_merged_axis(ax,data1,data2,c1,c2,format=None,**kwargs):


        make_violin_plot(ax,data1,**dict(kwargs,c=c1,column_type="merged_axis"))
        make_violin_plot(ax,data2,**dict(kwargs,format=format,c=c2,column_type="merged_axis"))


        min1 = np.min(data1)   
        min2 = np.min(data2)
        if min1 < min2:
            #text = f"{min1:.{kwargs['precision']}f}"
            #underlined_annotation(fig,ax,(.075,extremal_tick_ypos),text,linestyle=LRT_linestyle,ha="left",fontsize=8,alpha=.8)

            text = ax.annotate(f"{min1:.{kwargs['precision']}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8) 
            mark_as_algorithm(ax,text,T_color,T_linestyle,algorithm="TAME")
            #underline_text(ax,text,T_color,T_linestyle) 
            """
            tb = text.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
            ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),             xycoords="figure fraction",arrowprops=dict(arrowstyle="-", color='k',linestyle=T_linestyle))
            """
        else:
            text = ax.annotate(f"{min2:.{kwargs['precision']}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)  
            mark_as_algorithm(ax,text,LRT_color,LRT_linestyle,algorithm="LRTAME")
            #underline_text(ax,text,LRT_color,LRT_linestyle)

        #minimum_val = min([np.min(data1),np.min(data2)])
        maximum_val = min([np.max(data1),np.max(data2)])
        max1 = np.max(data1)   
        max2 = np.max(data2)
        if max1 > max2:
            text = f"{maximum_val:.{kwargs['precision']}f}"
            text = ax.annotate(f"{max1:.{kwargs['precision']}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)  
            mark_as_algorithm(ax,text,T_color,T_linestyle,algorithm="TAME")
            #underline_text(ax,text,T_color,T_linestyle)
            #underlined_annotation(fig,ax,(.925,extremal_tick_ypos),text,linestyle=LRT_linestyle,ha="right",fontsize=8,alpha=.8)
        else:
            text = ax.annotate(f"{maximum_val:.{kwargs['precision']}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
            mark_as_algorithm(ax,text,LRT_color,LRT_linestyle,algorithm="LRTAME")
            #underline_text(ax,text,LRT_color,LRT_linestyle)

        ax.annotate(f"{np.median(data1):.{kwargs['precision']}f}",xy=(.7,.5),xycoords="axes fraction",ha="center",fontsize=10)

        ax.annotate(f"{np.median(data2):.{kwargs['precision']}f}",xy=(.3,.5),xycoords="axes fraction",ha="center",fontsize=10)




        #for x in sorted(dir(text)):
        #    print(x)
        """
        if format is None:        
            ax.annotate(f"{np.median(data1):.{kwargs[:precision]}f}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data1):.{precision}f}",xy=(.075,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data1):.{precision}f}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        elif format == "scientific":
            ax.annotate(f"{np.median(data):.{precision}e}",xy=(.5,.2),xycoords="axes fraction",ha="center",fontsize=10)
            ax.annotate(f"{np.min(data):.{precision}e}",xy=(.025,extremal_tick_ypos),xycoords="axes fraction",ha="left",fontsize=8,alpha=.8)
            ax.annotate(f"{np.max(data):.{precision}e}",xy=(.925,extremal_tick_ypos),xycoords="axes fraction",ha="right",fontsize=8,alpha=.8)
        else:
            print(f"expecting format to be 'scientific' or None, got:{format}")
        """

    def make_violin_plot_legend(ax,c="k"):
        
        np.random.seed(12)#Ok looking:12
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.6], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()
        ax.set_ylim(.5,1.0)
        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("median",xy=(.5,.4),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(.025,-.125),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
        ax.annotate(f"max",xy=(.975,-.125),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.3)
                b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])

    def make_merged_violin_plot_legend(ax):
        
        np.random.seed(12)#Ok looking:12
        v1= ax.violinplot([np.random.normal(-.5,.25) for i in range(50)], points=100, positions=[.6], showmeans=False, showextrema=False, showmedians=True,widths=.6,vert=False)#,quantiles=[[.2,.8]]*len(data2))

        v2 = ax.violinplot([np.random.normal(1.0,.25) for i in range(50)], points=100, positions=[.6], 
                          showmeans=False, showextrema=False, showmedians=True,widths=.6,vert=False)
        ax.set_ylim(.5,1.0)

        for (c,v) in [(LRT_color,v1),(T_color,v2)]:
            #  --  update median lines to have a gap  --  #
            ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
            #newMedianLines = [[(x0,y0-.125),(x0,y0 + (y1-y0)/2 -.1)]]#,[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
            newMedianLines = [[(x0,y1 +.05),(x0,1.5)]]
            v["cmedians"].set_segments(newMedianLines)

            med_label1 = ax.annotate("LRT med.",xy=(.25,.35),xycoords="axes fraction",ha="center",va="center",fontsize=10)
            #mark_as_algorithm(ax,med_label1,LRT_color,LRT_linestyle,algorithm="LRTAME")
            med_label2 = ax.annotate("T med.",xy=(.7,.35),xycoords="axes fraction",ha="center",va="center",fontsize=10)
            #mark_as_algorithm(ax,med_label2,T_color,T_linestyle,algorithm="TAME")
            min_label = ax.annotate(f"min",xy=(.075,-.125),xycoords="axes fraction",ha="left",fontsize=9,alpha=.8)
            mark_as_algorithm(ax,min_label,LRT_color,LRT_linestyle,algorithm="LRTAME")
            
            max_label = ax.annotate(f"max",xy=(.925,-.125),xycoords="axes fraction",ha="right",fontsize=9,alpha=.8)
            mark_as_algorithm(ax,max_label,T_color,T_linestyle,algorithm="TAME")
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.3)
                b.set_color(c)
                m = np.mean(b.get_paths()[0].vertices[:, 1])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1],m, np.inf)

                def clip_to_top_of_violin(segment):
                    med_line_x = segment.get_paths()[0].vertices[0, 0]

                    # find the y-vals of the violin plots near the med-lines
                    distances = [abs(p[0]- med_line_x) for p in b.get_paths()[0].vertices]
                    k = 5
                    closest_x_points = np.argpartition(distances,k)[:k]
                    new_max_y = np.max([b.get_paths()[0].vertices[idx,1] for idx in closest_x_points] )
                    new_max_y += .02
                    #clip the lines 
                    segment.get_paths()[0].vertices[:, 1] = np.clip(segment.get_paths()[0].vertices[:, 1],-np.inf,new_max_y)


                clip_to_top_of_violin(v["cmedians"])



    

    #
    #  Parse Data
    #

    results_path = TAME_RESULTS + "RandomGraphTAME_dupNoise/TAME_clique_scaling_exps/"

    elemwise_list_sum =lambda l1,l2: [a + b for (a,b) in zip(l1,l2)]
    #filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[25]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json" 
    filename = "LowRankTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7,8]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"

    with open(results_path + filename,"r") as file:
        data = json.load(file)
        LR_results = {}
        debug_results = {}
        LR_seeds = {}

        for (order, LRTAME_output) in data:
            trials = len(LRTAME_output)
            LR_results[order] = {
                "full runtime":{},
                "contraction runtime":{},
                "A motifs":[],
                "ranks":{},
            }
            debug_results[order] = {
                "runtimes":[],
                "ranks":[]
            }
            LR_seeds[order] = []
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(LRTAME_output):
                LR_results[order]["A motifs"].append(A_motifs)
                LR_seeds[order].append(seed)
                for (params,profile) in profiling:
                    """
                    rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])
                    if params in LR_results[order]["full runtime"]:
                        LR_results[order]["full runtime"][params].append(rt)
                    else:
                        LR_results[order]["full runtime"][params] = [rt]
                    """
                    contract_rt = reduce(elemwise_list_sum,[
                        profile["low_rank_factoring_timings"],
                        profile["contraction_timings"],
                        ])
                    if params in LR_results[order]["contraction runtime"]:
                        LR_results[order]["contraction runtime"][params].append(np.max(contract_rt))
                    else:
                        LR_results[order]["contraction runtime"][params] = [np.max(contract_rt)]


                    if params == 'α:0.5_β:1.0':
                        debug_results[order]["ranks"].append(profile["ranks"])
                        debug_results[order]["runtimes"].append(contract_rt)

                    ranks = profile["ranks"]
                    if len(ranks) < 15:
                        # if algorithm terminated from tol bounds, extend last rank to fill the rest
                        ranks.extend([ranks[-1]]*(15-len(ranks)))

                    if params in LR_results[order]["ranks"]:
                        LR_results[order]["ranks"][params].append(profile["ranks"])
                    else:
                        LR_results[order]["ranks"][params] = [profile["ranks"]]

    #return debug_results

    file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[25]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    file = "TAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)

        T_results = {}


        for (order, TAME_output) in data:
            trials = len(TAME_output)
            T_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
 
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profiling) in enumerate(TAME_output):
                for (params,profile) in profiling:
                    """
                    rt = reduce(elemwise_list_sum,[
                        profile["contraction_timings"],
                        profile["matching_timings"],
                        profile["scoring_timings"],
                        ])

                    if params in T_results[order]:
                        T_results[order]["full runtime"][params].append(np.sum(rt))
                    else:
                        T_results[order]["full runtime"][params] = [np.sum(rt)]
                    """
                    if params in T_results[order]["contraction runtime"]:
                        T_results[order]["contraction runtime"][params].append(np.max(profile["contraction_timings"]))
                    else:
                        T_results[order]["contraction runtime"][params] = [np.max(profile["contraction_timings"])]


    file = "LambdaTAME_RandomGeometric_degreedist:LogNormal_alpha:[0.5,1.0]_beta:[0.0,1.0]_n:[100]_orders:[3,4,5,6,7]_p:0.5_sample:10000_sp:[0.2]_trials:25_results.json"
    with open(results_path + file,"r") as file:
        data = json.load(file)
        LT_results = {}

        all_shifts = [
            'α:0.5_β:0.0',
            'α:0.5_β:1.0',
            'α:1.0_β:0.0',
            'α:1.0_β:1.0',
        ]
        for (order, TAME_output) in data:
            trials = len(TAME_output)
            LT_results[order] = {
                "contraction runtime":{},
                "full runtime":{},
            }
            for trial_idx,(seed,p,n,sp,accuracy,dup_tol_accuracy,motifs_matched,A_motifs,B_motifs,A_motif_dist,B_motif_dist,profile) in enumerate(TAME_output):

                #  aggregate the runtimes
                full_rt = reduce(elemwise_list_sum,[
                        profile["Matching Timings"],
                        profile["TAME_timings"],
                        profile["Scoring Timings"],
                        ])

                for (i,rt) in enumerate(full_rt):
                    params = all_shifts[i]
                    if params in LT_results[order]["full runtime"]:
                        LT_results[order]["full runtime"][params].append(rt)
                    else:
                        LT_results[order]["full runtime"][params] = [rt]

                for (i,rt) in enumerate(profile["TAME_timings"]):
                    params = all_shifts[i]
                    if params in LT_results[order]["contraction runtime"]:
                        LT_results[order]["contraction runtime"][params].append(rt)
                    else:
                        LT_results[order]["contraction runtime"][params] = [rt]

    #
    #  Create Plots 
    #
    fig = plt.figure(figsize=(5,3))
    global_ax = plt.gca()
    global_ax.set_axis_off()


    parity = 1  #used for checkerboard effect in plots.             
    

    linestyles= {
        'α:0.5_β:0.0':"dotted",
        'α:0.5_β:1.0':"solid",
        'α:1.0_β:0.0':(0,(3,1,1,1)),
        'α:1.0_β:1.0':(0,(5,1))
    }

    # column assignment

    motif_label = 0
    A_motif_data = 1
    rank_data = 2
    TAME_rt = 3
    LRTAME_rt = 3
    LTAME_rt = 5

    #rt_data = 1

    main_gs = fig.add_gridspec(2, 1,hspace=0.0,wspace=0.0, height_ratios = [1,.15],
                                left=0.025,right=.975,top=.85,bottom=0.025)

    n = len(T_results.keys())
    m = 4 # - 1 for no LT 
          # - 1 for merging TAME w/ LRT
    
    heights = [.2,.3,.3,.7]
    row_height_ratios = [1]*n 
    #col_width_ratios.append(.8) 


    
    gs = main_gs[0].subgridspec(nrows=n,ncols=m,hspace=0.0,wspace=0.0, 
                          width_ratios=heights,height_ratios=row_height_ratios)
    legend_gs = main_gs[1].subgridspec(nrows=1,ncols=20)
                       
    gs_ax = np.empty((n,m),object)

    iterate_tick_idx = 2


    # -- plot runtime data -- # 
    motif_label_xloc = .4
    for idx,((LRT_order,LRT_data),(T_order,T_data),(LT_order,LT_data)) in enumerate(zip(LR_results.items(),T_results.items(),LT_results.items())):
        assert LRT_order == T_order
        assert LRT_order == LT_order

        order = LRT_order
        for col in range(m):

            """
            ax = fig.add_subplot(gs[idx,col])
            """

            if idx == 0:
                ax = fig.add_subplot(gs[idx,col])
            else:
                ax = fig.add_subplot(gs[idx,col],sharex=gs_ax[0,col])
            
            gs_ax[idx,col] = ax


            if col == motif_label:
                ax.annotate(f"{order}",xy=(motif_label_xloc, .375), xycoords='axes fraction', c="k",size=10,ha="center",va="center")#,weight="bold")
            elif col == TAME_rt:
                #ax.set_xscale("log")
                make_violin_plot_merged_axis(ax,T_data["contraction runtime"]['α:0.5_β:1.0'],
                                                LRT_data["contraction runtime"]['α:0.5_β:1.0'],
                                                T_color,LRT_color, precision=2,v_alpha=1.0,xscale="log")



                #make_violin_plot(ax,T_data["contraction runtime"]['α:0.5_β:1.0'],precision=2,c=T_color,v_alpha=1.0,xscale="log")
                #make_violin_plot(ax,LRT_data["contraction runtime"]['α:0.5_β:1.0'],precision=2,c=LRT_color,v_alpha=1.0,xscale="log")

            elif col == LRTAME_rt:
                #ax.set_xscale("log")
                print("test")
              
            elif col == LTAME_rt:
                make_violin_plot(ax,LT_data["contraction runtime"]['α:0.5_β:1.0'],
                                 precision=1,c=LT_color,v_alpha=.3,format = "scientific",xscale="log")
            elif col == A_motif_data:
                make_violin_plot(ax,LRT_data["A motifs"],precision=0,c="k",v_alpha=.2)

            elif col == rank_data:
                make_violin_plot(ax,np.array(LRT_data["ranks"]['α:0.5_β:1.0']).max(axis=1),precision=0,c="k",v_alpha=.2)
                #ax.set_xlim(13,61)

    vp_legend_ax = fig.add_subplot(legend_gs[1:6])
    make_violin_plot_legend(vp_legend_ax)

    merged_vp_legend_ax = fig.add_subplot(legend_gs[11:20])
    make_merged_violin_plot_legend(merged_vp_legend_ax)

    marker_legend_ax = fig.add_subplot(legend_gs[6:10])
    label_font = 11
    LRT_label = marker_legend_ax.annotate("LR-TAME",rotation=0,ha="right",va="top",xy=(.85, 1.0), xycoords='axes fraction',c=LRT_color,fontsize=label_font)
    mark_as_algorithm(marker_legend_ax,LRT_label,LRT_color,LRT_linestyle,"LRTAME")

    TAME_label = marker_legend_ax.annotate("TAME",rotation=0,ha="right",va="bottom",xy=(.85, 0.0), xycoords='axes fraction',c=T_color,fontsize=label_font)
    mark_as_algorithm(marker_legend_ax ,TAME_label,T_color,T_linestyle,"TAME")


    #make_violin_plot_legend(merged_vp_legend_ax)
    # -- create legends -- # 



    #
    #  Adjust Axes
    #
    
    # -- Set the column titles -- #
    title_ypos = 1.1 
    annotation_ypos = .6
    gs_ax[0,motif_label].annotate("Clique\nSize",ha="center",va="bottom",xy=(motif_label_xloc, title_ypos), xycoords='axes fraction')
    
    
 
    #bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=-.1)
    
    #gs_ax[0,LRTAME_rt].set_ylabel("Max Iterate\nTTV Time (s)",rotation=0,labelpad=30,ha="center",va="center")#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
    gs_ax[0,LRTAME_rt].annotate("Longest\nContraction (s)",rotation=0,ha="center",va="bottom",xy=(.5, title_ypos), xycoords='axes fraction')#,c=LT_color,xy=(.3, .125), xycoords='axes fraction').set_bbox(bbox)
   

    gs_ax[0,rank_data].annotate("Max (LR-)TAME\nIterate Rank",ha="center",va="bottom",xy=(.5, title_ypos), xycoords='axes fraction')

    gs_ax[0,A_motif_data].annotate("A motifs",xy=(.5, title_ypos), xycoords='axes fraction',ha="center",va="bottom")

    # -- make a label for shared x-axis
    super_title_ypos = .875


    additional_ax = [
        global_ax,vp_legend_ax,merged_vp_legend_ax,marker_legend_ax #legend_ax # annotation_ax,
    ]

    for ax in chain(gs_ax.reshape(-1),additional_ax):
        ax.set_yticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_xticklabels([])

    plt.show()



#
#  LVGNA Alignments
#
def make_LVGNA_runtime_performance_plots_v2():

    f = plt.figure(figsize=(12,4))#dpi=60)
    
    widths = [2.5,1.25,2.5]
    spec = f.add_gridspec(ncols=3, nrows=1, width_ratios=widths,hspace=.1,left=.05,right=.96,top=.95,bottom=.15)
    all_ax = np.empty(3,object)
    for i in range(3):
        all_ax[i] = f.add_subplot(spec[i])
        
    #f,axes = plt.subplots(1,3)


    #f.set_size_inches(11.5, 5.5)
    #f.set_size_inches(16, 10)

    height = 0.8
    width = 0.475 
    far_left = .08
    bottom = .125
    pad = .08

    #rectangle1 = [far_left, bottom, width, height]
    #rectangle2 = [far_left+width+pad, bottom, .3,height]

    #axes = [plt.axes(rectangle1),plt.axes(rectangle2)]

    make_LVGNA_runtime_plots_v2(all_ax[0])
    make_LVGNA_performance_plots_v2(all_ax[1])
    make_LVGNA_TTVMatchingRatio_runtime_plots_v2(all_ax[2])

    plt.show()

def make_LVGNA_performance_plots_v2(ax=None,mettix_version=False):

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

    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-16_profile:true_tol:1e-12_results.json","r") as f:    
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-12_results.json","r") as f:
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
        fig =plt.figure(figsize=(2.5,4))
        spec = fig.add_gridspec(nrows=1, ncols=1,left=.25,right=.95,top=.95,bottom=.125)
        ax = fig.add_subplot(spec[0,0]) 
        show_plot = True
    else:
        show_plot = False

    

   
    #for i = 1:size(c, 2)
    #plot!(t, [sum(R[:,i] .<= ti)/size(c,1) for ti in t], label="Alg $i", t=:steppre, lw=2)
    #end

    
    #ax = [ax] #jerry rigged
    ax.plot(range(len(TAME_performance)), TAME_performance, label="TAME", c=T_color,linestyle=T_linestyle)
    #bold_outlined_text(ax,"TAME (C++)",t1_color,(.82, .9))
    #ax.annotate("TAME (C++)",xy=(.55, .81), xycoords='axes fraction', c=t1_color,rotation=-18)
   

    ax.plot(range(len(LambdaTAME_performance)), LambdaTAME_performance, label="$\Lambda$-TAME - (rom)", c=darkest_t2_color,linestyle=LT_rom_linestyle)
    #ax.annotate("$\Lambda$-TAME-(rom)",xy=(.4, .78), xycoords='axes fraction', c=darkest_t2_color,rotation=-22.5)
    
    ax.plot(range(len(LambdaTAMEGramMatching_performance)), LambdaTAMEGramMatching_performance, label="$\Lambda$-TAME", c=LT_color,linestyle=LT_linestyle)
    #ax.annotate("$\Lambda$-TAME",xy=(.4, .96), xycoords='axes fraction', c=t2_color,rotation=0)
    
    
    ax.plot(range(len(LowRankTAME_performance)), LowRankTAME_performance, label="LowRankTAME", c=LRT_color,linestyle=LRT_linestyle)
    #ax.annotate("LowRankTAME",xy=(.65, .91), xycoords='axes fraction', c=t4_color)
    
    ax.plot(range(len(LowRankTAME_LRMatch_performance)), LowRankTAME_LRMatch_performance, label="LowRankTAME-(lrm)", c=LRT_lrm_color,linestyle=LRT_lrm_linestyle)
    #ax.annotate("LowRankTAME-(lrm)",xy=(.3, .7),xycoords='axes fraction', c=t6_color,rotation=-15)
    
    if not mettix_version:
        ax.plot(range(len(LowRankEigenAlign_performance)),LowRankEigenAlign_performance,label="LowRankEigenAlign", c=LREigenAlign_color,linestyle=LREigenAlign_linestyle)
        ax.plot(range(len(LGRAAL_performance)), LGRAAL_performance, label="LGRAAL", c=LGRAAL_color,linestyle=LGRAAL_linestyle)
    
        if show_plot:
            bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.05)
            ax.annotate("LowRankEigenAlign",xy=(.11, .018),xycoords='axes fraction', c=LREigenAlign_color,rotation=-40).set_bbox(bbox)
            ax.annotate("L-GRAAL",xy=(.47, .46), xycoords='axes fraction', c=LGRAAL_color)
        else:
            ax.annotate("LowRankEigenAlign",xy=(.125, .075),xycoords='axes fraction', c=LREigenAlign_color,rotation=-40)
            ax.annotate("L-GRAAL",xy=(.47, .46), xycoords='axes fraction', c=LGRAAL_color)
        
        

    
    ax.set_ylabel("performance ratio")
    ax.grid(which="both")
    ax.set_xlabel("sorted experiment rank")
    """
    ax[1].plot(range(len(old_TAME_accuracy)),old_TAME_accuracy,label="TAME", c=t5_color)
    ax[1].plot(range(len(new_TAME_accuracy)),new_TAME_accuracy, label="$\Lambda$-TAME", c=t2_color)
    ax[1].plot(range(len(LGRAAL_accuracy)),LGRAAL_accuracy,label="L-GRAAL", c=t1_color)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("experiment rank")
    ax[1].grid(which="both")
    """
    #plt.legend()
    if show_plot:
        plt.show()

def make_LVGNA_performance_plots_with_triangles(gs=None):

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

    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-16_profile:true_tol:1e-12_results.json","r") as f:    
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-12_results.json","r") as f:
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

        
        LGRAAL_accuracy.append(LGRAAL_tri_results[i,j])
        LGRAAL_ratio = LGRAAL_tri_results[i,j]/best
        LGRAAL_performance.append(LGRAAL_ratio)

        #TODO: refactor to TAME
        ratio =  Original_TAME_tri_results[i,j]/ best
        TAME_accuracy.append(Original_TAME_tri_results[i,j])
        TAME_performance.append(ratio)

        #TODO: refactor to LambdaTAME
        LambdaTAME_accuracy.append(LambdaTAME_results[i,j])
        ratio = LambdaTAME_results[i, j] / best
        LambdaTAME_performance.append(ratio)

        LambdaTAMEGramMatching_accuracy.append(LambdaTAMEGramMatching_results[i,j])
        ratio = LambdaTAMEGramMatching_results[i,j] / best
        LambdaTAMEGramMatching_performance.append(ratio)

        LowRankTAME_accuracy.append(LowRankTAME_results[i,j])
        ratio = LowRankTAME_results[i, j] / best
        LowRankTAME_performance.append(ratio)

        LowRankTAME_LRMatch_accuracy.append(LowRankTAME_LRMatch_results[i,j])
        ratio = LowRankTAME_LRMatch_results[i, j] / best
        LowRankTAME_LRMatch_performance.append(ratio)

        LowRankEigenAlign_accuracy.append(LowRankEigenAlign_results[i,j])
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

    if gs is None:
        fig = plt.figure(figsize=(2.5,4))
        gs = fig.add_gridspec(nrows=2, ncols=1,left=.25,right=.95,top=.95,bottom=.125)
        show_plot = True
    else:
        fig = plt.gcf()
        show_plot = False

    

   
    #for i = 1:size(c, 2)
    #plot!(t, [sum(R[:,i] .<= ti)/size(c,1) for ti in t], label="Alg $i", t=:steppre, lw=2)
    #end

    #
    #  --  plot the perfomance plots -- #
    #
    ax = fig.add_subplot(gs[0]) 

    ax.plot(range(len(TAME_performance)), TAME_performance, label="TAME", c=T_color,linestyle=T_linestyle)
    #ax.annotate("TAME (C++)",xy=(.55, .81), xycoords='axes fraction', c=t1_color,rotation=-18)
    ax.plot(range(len(LambdaTAME_performance)), LambdaTAME_performance, label="$\Lambda$-TAME - (rom)", c=darkest_t2_color,linestyle=LT_rom_linestyle)
    #ax.annotate("$\Lambda$-TAME-(rom)",xy=(.4, .78), xycoords='axes fraction', c=darkest_t2_color,rotation=-22.5)
    ax.plot(range(len(LambdaTAMEGramMatching_performance)), LambdaTAMEGramMatching_performance, label="$\Lambda$-TAME", c=LT_color,linestyle=LT_linestyle)
    #ax.annotate("$\Lambda$-TAME",xy=(.4, .96), xycoords='axes fraction', c=t2_color,rotation=0)
    ax.plot(range(len(LowRankTAME_performance)), sorted(LowRankTAME_accuracy,reverse=True), label="LowRankTAME", c=LRT_color,linestyle=LRT_linestyle)
    #ax.annotate("LowRankTAME",xy=(.65, .91), xycoords='axes fraction', c=t4_color)
    ax.plot(range(len(LowRankTAME_LRMatch_performance)), LowRankTAME_LRMatch_performance, label="LowRankTAME-(lrm)", c=LRT_lrm_color,linestyle=LRT_lrm_linestyle)
    #ax.annotate("LowRankTAME-(lrm)",xy=(.3, .7),xycoords='axes fraction', c=t6_color,rotation=-15)
    

    ax.plot(range(len(LowRankEigenAlign_performance)),LowRankEigenAlign_performance,label="LowRankEigenAlign", c=LREigenAlign_color,linestyle=LREigenAlign_linestyle)
    ax.plot(range(len(LGRAAL_performance)), LGRAAL_performance, label="LGRAAL", c=LGRAAL_color,linestyle=LGRAAL_linestyle)

    if show_plot:
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.05)
        ax.annotate("LowRankEigenAlign",xy=(.11, .018),xycoords='axes fraction', c=LREigenAlign_color,rotation=-40).set_bbox(bbox)
        ax.annotate("L-GRAAL",xy=(.47, .46), xycoords='axes fraction', c=LGRAAL_color)
    else:
        ax.annotate("LowRankEigenAlign",xy=(.125, .075),xycoords='axes fraction', c=LREigenAlign_color,rotation=-40)
        ax.annotate("L-GRAAL",xy=(.47, .46), xycoords='axes fraction', c=LGRAAL_color)
        
    ax.set_ylabel("performance ratio")
    ax.grid(which="both")
    ax.set_xlabel("sorted experiment rank")


    #
    #  --  plot the raw triangle matched -- #
    #
    ax = fig.add_subplot(gs[1]) 

    ax.plot(range(len(TAME_performance)), sorted(TAME_accuracy,reverse=True), label="TAME", c=T_color,linestyle=T_linestyle)
    #ax.annotate("TAME (C++)",xy=(.55, .81), xycoords='axes fraction', c=t1_color,rotation=-18)
    ax.plot(range(len(LambdaTAME_performance)), sorted(LambdaTAME_accuracy,reverse=True), label="$\Lambda$-TAME - (rom)", c=darkest_t2_color,linestyle=LT_rom_linestyle)
    #ax.annotate("$\Lambda$-TAME-(rom)",xy=(.4, .78), xycoords='axes fraction', c=darkest_t2_color,rotation=-22.5)
    ax.plot(range(len(LambdaTAMEGramMatching_performance)), sorted(LambdaTAMEGramMatching_accuracy,reverse=True), label="$\Lambda$-TAME", c=LT_color,linestyle=LT_linestyle)
    #ax.annotate("$\Lambda$-TAME",xy=(.4, .96), xycoords='axes fraction', c=t2_color,rotation=0)
    ax.plot(range(len(LowRankTAME_performance)), sorted(LowRankTAME_accuracy,reverse=True), label="LowRankTAME", c=LRT_color,linestyle=LRT_linestyle)
    #ax.annotate("LowRankTAME",xy=(.65, .91), xycoords='axes fraction', c=t4_color)
    ax.plot(range(len(LowRankTAME_LRMatch_performance)), sorted(LowRankTAME_LRMatch_accuracy,reverse=True), label="LowRankTAME-(lrm)", c=LRT_lrm_color,linestyle=LRT_lrm_linestyle)
    #ax.annotate("LowRankTAME-(lrm)",xy=(.3, .7),xycoords='axes fraction', c=t6_color,rotation=-15)
    
    ax.plot(range(len(LowRankEigenAlign_performance)),sorted(LowRankEigenAlign_accuracy,reverse=True),label="LowRankEigenAlign", c=LREigenAlign_color,linestyle=LREigenAlign_linestyle)
    ax.plot(range(len(LGRAAL_performance)), sorted(LGRAAL_accuracy,reverse=True), label="LGRAAL", c=LGRAAL_color,linestyle=LGRAAL_linestyle)

    if show_plot:
        plt.show()


def make_LVGNA_runtime_plots_v2(ax=None,mettix_version=False):

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

    def process_LambdaTAMEData(exp_results):

        exp_idx = {name:i for i,name in enumerate(graph_names)}
        runtime_data = np.zeros((len(graph_names),len(graph_names)))
        #runtime_data = -np.ones((len(graph_names),len(graph_names)))
        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            #print(runtime.keys())
            contraction_rt = sum(runtime['TAME_timings'])
            matching_rt = sum(runtime['Matching Timings'])
            scoring_rt = sum(runtime['Scoring Timings'])
            #print(contraction_rt,matching_rt,scoring_rt)

            runtime_data[i,j] = matching_rt + contraction_rt + scoring_rt 
            runtime_data[j,i] = matching_rt + contraction_rt + scoring_rt 

        return runtime_data

    #with open(TAME_RESULTS + "LVGNA_Experiments/LambdaTAME_LVGNA_results_alphas:[.5,1.0]_betas:[0,1e0,1e1,1e2]_iter:15.json","r") as f:
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:rankOneMatching_profile:true_tol:1e-6_results.json","r") as f:
    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:rankOneMatching_profile:true_tol:1e-6_results.json")
        LambdaTAMERankOneMatching_runtimes = process_LambdaTAMEData(json.load(f))

    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-6_results.json","r") as f:  
    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-8_profile:true_tol:1e-6_results.json","r") as f: 
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwiseAlignment_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-8_profile:true_tol:1e-6_results.json","r") as f: 
        LambdaTAMEGramMatching_runtimes = process_LambdaTAMEData(json.load(f))
        #return json.load(f)

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
    LambdaTAMERankOneMatching_exp_runtimes = []
    LambdaTAMEGramMatching_exp_runtimes = []
    LowRankTAME_exp_runtimes = []
    LowRankTAME_LRM_exp_runtimes = []
    LowRankEigenAlign_exp_runtimes = []

    Is,Js = np.triu_indices(n,k=1)
    for i,j in zip(Is,Js):

        LGRAAL_exp_runtimes.append(LGRAAL_runtimes[i,j])
        TAME_exp_runtimes.append(TAME_runtimes[i,j])
        LambdaTAMERankOneMatching_exp_runtimes.append(LambdaTAMERankOneMatching_runtimes[i,j])
        LambdaTAMEGramMatching_exp_runtimes.append(LambdaTAMEGramMatching_runtimes[i,j])
        LowRankTAME_exp_runtimes.append(LowRankTAME_runtimes[i, j])
        LowRankTAME_LRM_exp_runtimes.append(LowRankTAME_LRM_runtimes[i, j])
        LowRankEigenAlign_exp_runtimes.append(LowRankEigenAlign_runtimes[i,j])

        problem_sizes.append(triangle_counts[graph_names[i]]*triangle_counts[graph_names[j]])

    #
    #  Plot results
    #


    if ax is None:
        fig =plt.figure(figsize=(5,4))
        spec = fig.add_gridspec(nrows=1, ncols=1,left=.125,right=.95,top=.95,bottom=.125)
        ax = fig.add_subplot(spec[0,0]) 
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
    ax.scatter(problem_sizes,TAME_exp_runtimes,label="TAME", c=T_color,marker='s')
    plot_1d_loess_smoothing(problem_sizes,TAME_exp_runtimes,loess_smoothing_frac,ax,c=T_color,linestyle=T_linestyle)
    #ax[0].plot(range(len(old_TAME_performance)), old_TAME_performance, label="TAME", c=t4_color)
    ax.annotate("TAME (C++)",xy=(.53, .85), xycoords='axes fraction', c=T_color)

    
    #print(new_TAME_exp_runtimes)
    ax.scatter(problem_sizes,LambdaTAMERankOneMatching_exp_runtimes,label="$\Lambda$-TAME-(rom)",facecolors='none', edgecolors=LT_rom_color,marker='^')
    plot_1d_loess_smoothing(problem_sizes,LambdaTAMERankOneMatching_exp_runtimes,loess_smoothing_frac,ax,c=LT_rom_color,linestyle=LT_rom_linestyle)
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    ax.annotate("$\Lambda$-TAME\n(rom)",xy=(.6, .09), xycoords='axes fraction', c=LT_rom_color,ha="center")
    
    ax.scatter(problem_sizes,LambdaTAMEGramMatching_exp_runtimes,label="$\Lambda$-TAME", c=LT_color ,marker='^')
    plot_1d_loess_smoothing(problem_sizes,LambdaTAMEGramMatching_exp_runtimes,loess_smoothing_frac,ax,c=LT_color ,linestyle=LT_linestyle)
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    ax.annotate("$\Lambda$-TAME",xy=(.84, .55), xycoords='axes fraction', c=LT_color,rotation=20)

    ax.scatter(problem_sizes,LowRankTAME_exp_runtimes,label="LowRankTAME", c=LRT_color)
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_exp_runtimes,loess_smoothing_frac,ax,c=LRT_color,linestyle=LRT_linestyle)
    ax.annotate("LowRank\nTAME",xy=(.8, .7), xycoords='axes fraction', c=LRT_color,ha="left",rotation=40)
 
    ax.scatter(problem_sizes,LowRankTAME_LRM_exp_runtimes,facecolors='none',edgecolors=LRT_lrm_color,label="LowRankTAME-(lrm)")
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_LRM_exp_runtimes,loess_smoothing_frac,ax,c=LRT_lrm_color,linestyle=LRT_lrm_linestyle)
    ax.annotate("LowRank\nTAME-(lrm)",xy=(.775, .43), xycoords='axes fraction', c=LRT_lrm_color,ha="left",rotation=15)
 
    if not mettix_version:
        ax.scatter(problem_sizes,LowRankEigenAlign_exp_runtimes,label="LowRankEigenAlign",c=LREigenAlign_color,marker="*")
        plot_1d_loess_smoothing(problem_sizes,LowRankEigenAlign_exp_runtimes,loess_smoothing_frac,ax,c=LREigenAlign_color,linestyle=LREigenAlign_linestyle)
        ax.annotate("LowRank\nEigenAlign",xy=(.79, .15), xycoords='axes fraction', c=LREigenAlign_color,ha="left",rotation=30)
        
        ax.scatter(problem_sizes, LGRAAL_exp_runtimes,label="$LGRAAL", c=LGRAAL_color,zorder=-1,marker='x')
        plot_1d_loess_smoothing(problem_sizes,LGRAAL_exp_runtimes,loess_smoothing_frac,ax,c=LGRAAL_color,linestyle=LGRAAL_linestyle)
        #ax.plot(range(len(LGRAAL_performance)), LGRAAL_performance, label="LGRAAL", c=t1_color)
        ax.annotate("L-GRAAL",xy=(.2, .7), xycoords='axes fraction', c=LGRAAL_color)

    

    """
    ax[1].plot(range(len(old_TAME_accuracy)),old_TAME_accuracy,label="TAME", c=t5_color)
    ax[1].plot(range(len(new_TAME_accuracy)),new_TAME_accuracy, label="$\Lambda$-TAME", c=t2_color)
    ax[1].plot(range(len(LGRAAL_accuracy)),LGRAAL_accuracy,label="LGRAAL", c=t1_color)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("experiment rank")
    ax[1].grid(which="both")
    """
    #plt.legend()
    #plt.tight_layout()

    ax.set_yscale('symlog')

    if show_plot:
        plt.show()

def make_LVGNA_TTVMatchingRatio_runtime_plots_v2(ax=None):
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
        

    def parseLambdaTAMEData(exp_results):

        exp_idx = {name:i for i,name in enumerate(graph_names)}
        ratio_data = np.zeros((len(graph_names),len(graph_names)))

        for (file_A,file_B,matched_tris,max_tris,_,runtime) in exp_results:
            graph_A = " ".join(file_A.split(".ssten")[0].split("_"))
            graph_B = " ".join(file_B.split(".ssten")[0].split("_"))
            i = exp_idx[graph_A]
            j = exp_idx[graph_B]

            contraction_rt = sum(runtime['TAME_timings'])
            matching_rt = sum(runtime['Matching Timings'])

            ratio_data[j,i] = matching_rt/contraction_rt
            ratio_data[i,j] = matching_rt/contraction_rt

        return ratio_data

    #with open(TAME_RESULTS + "LVGNA_Experiments/LambdaTAME_LVGNA_results_alphas:[.5,1.0]_betas:[0,1e0,1e1,1e2]_iter:15.json","r") as f:
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:rankOneMatching_profile:true_tol:1e-6_results.json","r") as f:
        LambdaTAME_ratio = parseLambdaTAMEData(json.load(f))


  
    #with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwairAlignemnt_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_profile:true_tol:1e-6_results.json","r") as f:  
    with open(TAME_RESULTS + "LVGNA_Experiments/LVGNA_pairwiseAlignment_LambdaTAME_alphas:[.5,1.0]_betas:[0.0,1.0,10.0,100.0]_iter:15_matchingMethod:GramMatching_MatchingTol:1e-8_profile:true_tol:1e-6_results.json","r") as f: 
        LambdaTAMEGramMatching_ratio = parseLambdaTAMEData(json.load(f))


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
        fig =plt.figure(figsize=(5,4))
        spec = fig.add_gridspec(nrows=1, ncols=1,left=.175,right=.9,top=.95,bottom=.125)
        ax = fig.add_subplot(spec[0,0])
        show_plot = True
    else:
        show_plot = False
    #ax = [ax] #jerry rigged

    ax.set_ylim(1e-3,1e5)
    ax.set_xlim(2e5,5e11)

    label_font_size = None

    if show_plot:
        ax.set_ylabel("matching runtime\ncontraction runtime",fontsize=label_font_size)
        x_loc = -.17
        ax.annotate('', xy=(x_loc, .29), xycoords='axes fraction', xytext=(x_loc, 0.71),
                    arrowprops=dict(arrowstyle="-", color='k'))
    else:
        ax.set_ylabel("matching runtime / contraction runtime")

    ax.set_xlabel(r"|$T_A$||$T_B$|")
    #left axis labels
  
    ax.set_xscale("log")
    ax.set_yscale("log")
    loess_smoothing_frac = .3
    ax.grid(which="major",zorder=-2)
    ax.axhspan(1e-5,1,alpha=.1,color="k")
    
    ax.scatter(problem_sizes,TAME_exp_ratios,label="TAME", c=T_color,marker='s')
    plot_1d_loess_smoothing(problem_sizes,TAME_exp_ratios,loess_smoothing_frac,ax,c=T_color,linestyle=T_linestyle,logFilter=True)
    #ax[0].plot(range(len(old_TAME_performance)), old_TAME_performance, label="TAME", c=t4_color)
    ax.annotate("TAME (C++)",xy=(.1, .55), xycoords='axes fraction', c=T_color)

 
    ax.scatter(problem_sizes,LowRankTAME_exp_ratios,label="LowRankTAME", c=LRT_color)
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_exp_ratios,loess_smoothing_frac,ax,c=LRT_color,linestyle=LRT_linestyle,logFilter=True)
    ax.annotate("LowRank\nTAME",xy=(.01, .38), xycoords='axes fraction', c=LRT_color,ha="left")
 
    ax.scatter(problem_sizes,LowRankTAME_LRM_exp_ratios,facecolors='none',edgecolors=LRT_lrm_color,label="LowRankTAME-(lrm)")
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_LRM_exp_ratios,loess_smoothing_frac,ax,c=LRT_lrm_color,linestyle=LRT_lrm_linestyle,logFilter=True)
    ax.annotate("LowRankTAME-(lrm)",xy=(.1, .05), xycoords='axes fraction', c=LRT_lrm_color)
     
  
    ax.scatter(problem_sizes,LambdaTAMEGramMatching_exp_ratios,label="$\Lambda$-TAME", c=LT_color ,marker='^')
    plot_1d_loess_smoothing(problem_sizes,LambdaTAMEGramMatching_exp_ratios,loess_smoothing_frac,ax,c=LT_color ,linestyle=LT_linestyle)
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    ax.annotate("$\Lambda$-TAME",xy=(.1, .825), xycoords='axes fraction', c=LT_color)

    #print(new_TAME_exp_runtimes)
    ax.scatter(problem_sizes,LambdaTAME_exp_ratios,label="$\Lambda$-TAME-(rom)",facecolors='none', edgecolors=LT_rom_color,marker='^')
    plot_1d_loess_smoothing(problem_sizes,LambdaTAME_exp_ratios,loess_smoothing_frac,ax,c=LT_rom_color,linestyle=LT_rom_linestyle,logFilter=True)
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    ax.annotate("$\Lambda$-TAME\n(rom)",xy=(.25, .225), xycoords='axes fraction', ha="center",c=LT_rom_color)
    

    #ax.text(.95, .6,"contraction\ndominates")#, ha="center",c="k",rotation=90)
    #ax.text(.95, .1,"contraction\ndominates")#, ha="center",c="k",rotation=90)

    ax.annotate("matching\ndominates",xy=(1.05, .6), xycoords='axes fraction', ha="center",c="k",rotation=90,fontsize=label_font_size)
    ax.annotate("contraction\ndominates",xy=(1.05, .1), xycoords='axes fraction', ha="center",c="k",rotation=90,fontsize=label_font_size)
    
    if show_plot:
        plt.show()


def LVGNA_preProcessing():

    fig = plt.figure()
    global_ax = plt.gca()
    global_ax.set_axis_off()
    width_ratios = [1.0,.6]
    main_gs = fig.add_gridspec(1, 2,hspace=0.0,wspace=0.0, width_ratios = width_ratios,
                                left=0.025,right=.975,top=.85,bottom=0.025)

    
    runtime_gs = main_gs[0].subgridspec(nrows=2,ncols=1,hspace=0.0,wspace=0.0)
    matching_gs = main_gs[1].subgridspec(nrows=2,ncols=1,hspace=0.0,wspace=0.0)
    
    
    #preP_triangle_match_gs = main_gs[1].subgridspec(nrows=1,ncols=6)
    #postP_gs = main_gs[2].subgridspec(nrows=3,ncols=4)

    #initialize_all ax

    all_axes = np.empty((2,2),object)
    for i in range(0,1):
        all_axes[i,0] = fig.add_subplot(main_gs[i])

    make_LVGNA_TTVMatchingRatio_runtime_plots_v2(all_axes[0,0])
    make_LVGNA_runtime_plots_v2(all_axes[1,0])



    all_axes[0,1] = fig.add_subplot(gs[1:3,1])
    all_axes[1,1] = fig.add_subplot(gs[3:5,1])
            #TODO: fix this indexing. 

 

    preP_runtime_ax = np.empty((2),object)
    for i in range(2):
        preP_runtime_ax[i] = fig.add_subplot()

    preP_triangle_match_ax = np.empty((2),object)
    fig.add_subplot(legend_gs[1:6])

    #postP_ax = np.empty((3,2),object)
    

    
    #gs_ax = np.empty((3,10),object)
    plt.show()

    pass




#
#   Random Graph Experiment Plots 
#


def RandomGeometricDupNoise_KlauPostProcessing():

    #plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}')
    #plt.rc('text', usetex = True)
    #plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
   
    data_location = TAME_RESULTS + "klauExps/"
    fig= plt.figure(figsize=(5.75,4))
    n = 2
    m = 3
    spec = fig.add_gridspec(nrows=n, ncols=m,hspace=0.05,wspace=0.1,left=.075,right=.9,top=.975,bottom=.2)
    all_ax = []
    all_ax_gs = np.empty((n,m),object)
    #f, axes = plt.subplots(2,3)

    def process_data(data,version="old"):

        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {n:i for (i,n) in enumerate(sorted(set([datum[2] for datum in data])))}
        sp_idx = {sp:i for (i,sp) in enumerate(sorted(set([datum[3] for datum in data])))}

        trials = int(len(data)/(len(p_idx)*len(n_idx)*len(sp_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        klauAccuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx),len(sp_idx)),int)
        print(f"p vals: {p_idx}")
        print(f"n vals: {n_idx}")
        print(f"sp vals: {sp_idx}")


        for datum in data:
            
            if version == "old":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "new klau":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fstatus) = datum
            else:
                print(f"only supports 'old' and 'new klau', got {version}")

            i = p_idx[p]
            j = n_idx[n]
            k = sp_idx[sp]
            accuracy[i,j,k,trial_idx[i,j,k]] = acc
            klauAccuracy[i,j,k,trial_idx[i,j,k]] = klau_acc
            triMatch[i,j,k,trial_idx[i,j,k]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,k,trial_idx[i,j,k]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j,k] += 1
        
        return accuracy, klauAccuracy, triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx  

    def make_percentile_plot(plot_ax, x_domain,data,color,hatch=None,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.05,color)
        ]
        
        #plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)
        n,m = data.shape
        percentile_linewidth=.01

        for (lower_percentile,upper_percentile,alpha,color) in ribbons:
            #plot_ax.plot(np.percentile(data.T, lower_percentile, axis=0),c=color,linewidth=percentile_linewidth)
            #plot_ax.plot(np.percentile(data.T, upper_percentile, axis=0),c=color,linewidth=percentile_linewidth)

            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor=color,alpha=.1,edgecolor=color)
            """
            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor="None",hatch=hatch,edgecolor=color,alpha=.4)
            """

        for (col_func,alpha,color) in lines:
            line_data = []

            for i in range(n):
                line_data.append(col_func(data[i,:]))
        
            plot_ax.plot(x_domain,line_data,alpha=alpha,c=color,**kwargs)

   

    #hatches 
    LRT_Klau_hatch = None#"+"
    LT_Klau_hatch = None#"x"
    LRT_Tabu_hatch = None#"+"
    LT_Tabu_hatch = None#"+"
    
    

    default_p = .5
    default_n = 250
    default_sp = .25

    files = [
        "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.75]_sp:[0.2,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json",
    #    "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json",
    ]

    #
    #   p_edge experiments
    #
    sub_ax = [fig.add_subplot(spec[i,0]) for i in [0,1]]
    all_ax_gs[:,0] = sub_ax
    all_ax.append(sub_ax)
    #sub_ax = axes[:,0]
    #for file in files:
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))

    LT_file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + LT_file,'r') as f:
        _,  LT_TabuAccuracy,_,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    LRT_file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + LRT_file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy,LRT_triMatch,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy,_,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy,_,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")


    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy, _ ,LREA_TabuTriMatch, _,_,_ = process_data(json.load(f))

    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy, LREA_TriMatch ,LREA_KlauTriMatch, _,_,_ = process_data(json.load(f),"new klau")


    p_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_hatch,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_hatch,LT_Tabu_linestyle),
        
        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_hatch,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_hatch,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch, LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_TabuAccuracy, LREA_TabuTriMatch, LREA_Tabu_color,None,LREA_Tabu_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch, LREA_Klau_color,None,LREA_Klau_linestyle),
        #(LREA_KlauAccuracy)
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
    ]

    for (acc, tri, c,hatch,linestyle) in p_exps:
        make_percentile_plot(sub_ax[0],p_idx.keys(),acc[:,n_idx[default_n],sp_idx[default_sp],:],c,hatch,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],p_idx.keys(),tri[:,n_idx[default_n],sp_idx[default_sp],:],c,hatch,linestyle=linestyle)

    for ax in sub_ax:
        ax.set_xticks([0.0,.25,.5,.75,1.0])
        #ax.set_xlim(min(p_idx.keys()),max(p_idx.keys()))

    sub_ax[1].set_xticklabels(["0.0",".25",r"${\bf .5}$",".75","1.0"],rotation=45)
    sub_ax[1].set_xlabel(r"$p_{edge}$")

 
    #
    #   n size experiments
    #
    sub_ax = [fig.add_subplot(spec[i,1]) for i in [0,1]]

    all_ax.append(sub_ax)
    all_ax_gs[:,1] = sub_ax
    #sub_ax = axes[:,1]
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.75]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json' 
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy,_,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy,LRT_triMatch,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy,_,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")
    
    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy,_,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")
    
    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy,LREA_TriMatch,LREA_KlauTriMatch, _, _, _ = process_data(json.load(f),"new klau")


    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy,_,LREA_TabuTriMatch, _, _, _ = process_data(json.load(f))

    n_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_hatch,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_hatch,LT_Tabu_linestyle),
        
        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_hatch,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_hatch,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
        (LREA_TabuAccuracy,LREA_TabuTriMatch,LREA_Tabu_color,None,LREA_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRT_lrm_Klau_color),
    ]

    for (acc, tri, c,hatch,linestyle) in n_exps:
        make_percentile_plot(sub_ax[0],n_idx.keys(),acc[p_idx[default_p],:,sp_idx[default_sp],:],c,hatch,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],n_idx.keys(),tri[p_idx[default_p],:,sp_idx[default_sp],:],c,hatch,linestyle=linestyle)

    for ax in sub_ax:
        #ax.set_xlim(min(n_idx.keys()),max(n_idx.keys()))
        ax.set_xticks([100, 250, 500, 1000, 1250, 1500])
        ax.set_xlim(25,1575)

    
    sub_ax[1].set_xticklabels(["100",r"${\bf 250}$","500","1000","1250","1500"],rotation=60)
    sub_ax[1].set_xlabel(r"$|V_A|$")

 


    #
    #   step percentage experiments
    #
    # sub_ax = axes[:,2]
    sub_ax = [fig.add_subplot(spec[i,2]) for i in [0,1]]
    all_ax.append(sub_ax)
    all_ax_gs[:,2] = sub_ax
    #for file in files:
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.75]_sp:[0.2,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.75]_sp:[0.05,0.1,0.2,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))


    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json'
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy, _ ,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy, LRT_triMatch ,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f))

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy, _ ,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),"new klau")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
       #return json.load(f)
        _,  LRT_lrm_KlauAccuracy, _ ,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_data(json.load(f),version="new klau")

    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_TabuAccuracy, LREA_TriMatch ,LREA_TabuTriMatch,_,_,_ = process_data(json.load(f))
    
    file = "LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[250]_noiseModel:Duplication_p:[0.5]_sp:[0.05,0.1,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_KlauAccuracy, _ ,LREA_KlauTriMatch,p_idx2, n_idx2, sp_idx2 = process_data(json.load(f),"new klau")

    sp_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_hatch,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_hatch,LT_Tabu_linestyle),

        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_hatch,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_hatch,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_TabuAccuracy, LREA_TabuTriMatch,LREA_Tabu_color,None,LREA_Tabu_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
        #(LRT_lrm_KlauAccuracy, LRT_lrm_KlauTriMatch,LRT_lrm_Klau_color)
    ]
    

    for (acc, tri, c,hatch,linestyle) in sp_exps:
        make_percentile_plot(sub_ax[0],sp_idx.keys(),acc[p_idx[default_p],n_idx[default_n],:,:],c,hatch,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],sp_idx.keys(),tri[p_idx[default_p],n_idx[default_n],:,:],c,hatch,linestyle=linestyle)
  
    for ax in sub_ax:
        ax.set_xticks([.05,.1,.25,.5])
        ax.set_xlim(.025,.525)
        
    sub_ax[1].set_xticklabels(["5%","10%",r"${\bf 25\%}$","50%"],rotation=50)#52.5
    
    shift = -.25
    sub_ax[1].annotate(r"$|V_B|-|V_A|$",xy=(.5,-.09+shift),xycoords='axes fraction',ha="center")
    sub_ax[1].annotate('', xy=(.23, -.125+shift), xycoords='axes fraction', xytext=(.77, -.125+shift),
                       arrowprops=dict(arrowstyle="-", color='k',linewidth=.5))
    sub_ax[1].annotate(r"$|V_A|$",xy=(.5,-.21+shift),xycoords='axes fraction',ha="center")
    sub_ax[1].annotate(r"(%)",xy=(.78, -.125+shift),xycoords='axes fraction',ha="left",va="center")
    #sub_ax[1].set_xlabel(r"$\frac{|V_B|-|V_A|}{|V_A|}(\%)$")
   

    #
    #  Final Axes touch up 
    #

    all_ax_gs[0,0].set_ylabel("accuracy")
    all_ax_gs[1,0].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=-2.5)
    xloc = -.14
    all_ax_gs[1,0].annotate('', xy=(xloc , .2), xycoords='axes fraction', xytext=(xloc, 0.8),
                arrowprops=dict(arrowstyle="-", color='k'))
    for ax in all_ax_gs[0,:]:
        #ax.set_zorder(5)
        #ax.tick_params(axis="x",pad=-1)
        ax.set_xticklabels([])

    for ax in all_ax_gs[1,:]:
        ax.tick_params(axis="x",direction="out",pad=1)
        

    for ax in all_ax_gs[:,1]:
        ax.set_yticklabels([])
    all_ax_gs[0,0].set_yticklabels([])
    all_ax_gs[1,0].set_yticklabels([])
    #all_ax_gs[1,2].set_yticklabels([])
    #all_ax_gs[1,2].yaxis.set_label_position("right")


    for ax in all_ax_gs[:,2]:
        ax.yaxis.tick_right()

    for ax in all_ax_gs.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_ylim(-.075,1.075)
        ax.grid(True)

    all_ax_gs[1,1].annotate(r"$\Lambda$-TAME",xy=(.5,.275),color=LT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-10)
    all_ax_gs[0,2].annotate(r"$\Lambda$T" +"-Klau",xy=(.65,.875),color=LT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-17.5)
    all_ax_gs[1,2].annotate(r"$\Lambda$T" +"-Tabu",xy=(.775,.9),color=LT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-5)

    all_ax_gs[1,1].annotate("LRT-TAME",xy=(.325,.1),color=LRT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-10)
    all_ax_gs[1,1].annotate(r"LRT-Tabu",xy=(.7,.85),color=LRT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10)
    all_ax_gs[1,2].annotate(r"LRT-Klau",xy=(.45,.55),color=LRT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-40)



    #checkboard_color = [.925]*3
    """
    parity = 1
    for i in range(n):
        for j in range(m):  
            if parity == 1:
                all_ax_gs[i,j].patch.set_facecolor(checkboard_color)
                #all_gs_ax[i,j].patch.set_alpha(1.1)
        parity *= -1
    """

    plt.tight_layout()
    plt.show()

def RandomGeometricERNoise_KlauPostProcessing():

    data_location = TAME_RESULTS + "klauExps/"
    fig = plt.figure(figsize=(4,4))

    #fig, axes = plt.subplots(2,2,figsize=(4.25,5))
    n = 2
    m = 2
    spec = fig.add_gridspec(nrows=n, ncols=m,hspace=0.1,wspace=0.15,left=.1,right=.85,top=.95,bottom=.175)
    all_ax = []
    axes = np.empty((n,m),object)
    for i in range(n):
        for j in range(m):
            ax = fig.add_subplot(spec[i,j])
            axes[i,j] = ax


    def process_data(data,version="klau_old"):
        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {p:i for (i,p) in enumerate(sorted(set([datum[2] for datum in data])))}
        trials = int(len(data)/(len(p_idx)*len(n_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauAccuracy = np.zeros((len(p_idx),len(n_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx)),int)


        for datum in data:

            if version == "klau_new":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fbounds) = datum
            elif version == "klau_old":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "tabu":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            else:
                raise ValueError("only supports ")
            i = p_idx[p]
            j = n_idx[n]

            accuracy[i,j,trial_idx[i,j]] = acc
            LT_klauAccuracy[i,j,trial_idx[i,j]] = klau_acc
            triMatch[i,j,trial_idx[i,j]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,trial_idx[i,j]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j] += 1
        
        return accuracy, LT_klauAccuracy, triMatch, LT_klauTriMatch, p_idx, n_idx 

    def make_percentile_plot(plot_ax, x_domain,data,color,hatch=None,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.05,color)
        ]
        
        #plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)
        n,m = data.shape
        percentile_linewidth=.01

        for (lower_percentile,upper_percentile,alpha,color) in ribbons:
            #plot_ax.plot(np.percentile(data.T, lower_percentile, axis=0),c=color,linewidth=percentile_linewidth)
            #plot_ax.plot(np.percentile(data.T, upper_percentile, axis=0),c=color,linewidth=percentile_linewidth)

            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor=color,alpha=.1,edgecolor=color)
            """
            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor="None",hatch=hatch,edgecolor=color,alpha=.4)
            """

        for (col_func,alpha,color) in lines:
            line_data = []

            for i in range(n):
                line_data.append(col_func(data[i,:]))
        
            plot_ax.plot(x_domain,line_data,alpha=alpha,c=color,**kwargs)

    """
    LT_color = t2_color
    LRT_color = t4_color 
    LT_Klau_color = t3_color
    LT_Tabu_color = t6_color
    LRT_Klau_color = t5_color
    LRM_lrm_Klau_color = "k"
    LRT_Tabu_color = t1_color
    """


    #
    #   p_remove experiments
    #
    sub_ax = axes[:,0]

    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.0,0.005,0.01,0.05,0.1,0.2]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        accuracy, LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    files=[
    #    "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.0,0.005,0.01,0.05,0.1,0.2]_postProcess:TabuSearch_trialcount:20.json",
        "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.05,0.15,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json",
        "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.1,0.2]_postProcess:TabuSearch_trialcount:20.json"
    ]

    res = []
    for file in files:
        with open(data_location + file,'r') as f:
            res.extend(json.load(f))
    
    _,  LT_TabuAccuracy, _, LT_TabuTriMatch, p_idx, n_idx = process_data(res,"tabu")


    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        LRT_accuracy,  LRT_KlauAccuracy, LRT_triMatch, LRT_KlauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LRT_TabuAccuracy,_,LRT_TabuTriMatch, p_idx, n_idx = process_data(json.load(f),"tabu")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[250]_noiseModel:ErdosReyni_p:[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LRT_lrm_klauAccuracy,_,LRT_lrm_klauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")



    #make_percentile_plot(sub_ax[0],p_idx.keys(),LT_TabuAccuracy[:,0,:],LT_Tabu_color)
    #make_percentile_plot(sub_ax[1],p_idx.keys(),LT_TabuTriMatch[:,0,:],LT_Tabu_color)

    p_exps = [
        (accuracy,triMatch,LT_color,LT_linestyle),
        (LRT_accuracy,LRT_triMatch,LRT_color,LRT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_linestyle),
        #(LRT_lrm_klauAccuracy,LRT_lrm_klauTriMatch,LRM_lrm_Klau_color),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_linestyle),
    ]

    for (acc,tri,c,linestyle) in p_exps:
        make_percentile_plot(sub_ax[0],p_idx.keys(),acc[:,0,:],c,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],p_idx.keys(),tri[:,0,:],c,linestyle=linestyle)

    """
    make_percentile_plot(sub_ax[0],p_idx.keys(),accuracy[:,0,:],LT_color)
    make_percentile_plot(sub_ax[0],p_idx.keys(),LT_klauAccuracy[:,0,:],LT_Klau_color)
    make_percentile_plot(sub_ax[1],p_idx.keys(),triMatch[:,0,:],LT_color)
    make_percentile_plot(sub_ax[1],p_idx.keys(),LT_klauTriMatch[:,0,:],LT_Klau_color)
    """


    for ax in sub_ax:
        ax.set_xticks([0.01, 0.05, 0.1, 0.2,.3,.4])
        ax.set_xlim(-.05,.35)
        

    #sub_ax[1].set_xticklabels(sub_ax[1].get_xticks(),rotation=52.5)
    
    sub_ax[1].set_xticklabels([1.e-02, r"${\bf 0.05}$", .1, .2,.3,.4],rotation=50)

    sub_ax[1].set_xlabel(r"$p_{remove} \equiv  p$"+'\n'+r"$p_{add}=\frac{p\rho}{1-p}$",ha="center",labelpad=-5)

    #
    #   n size experiments
    #

    sub_ax = axes[:,1]
    #file="LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.01]_postProcess:KlauAlgo_trialcount:20.json"
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json'
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    #file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.01]_postProcess:TabuSearch_trialcount:20.json'
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json'
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _,  LT_TabuAccuracy, _, LT_TabuTriMatch, p_idx, n_idx = process_data(json.load(f),"tabu")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        LRT_accuracy,  LRT_KlauAccuracy, LRT_triMatch, LRT_KlauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _, LRT_TabuAccuracy,_,LRT_TabuTriMatch, p_idx, n_idx = process_data(json.load(f),"tabu")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        #return json.load(f)
        _,  LRT_lrm_KlauAccuracy, _, LRT_lrm_KlauTriMatch, p_idx, n_idx = process_data(json.load(f),"klau_new")



    """
    make_percentile_plot(sub_ax[0],n_idx.keys(),accuracy[0,:,:],LT_color)
    make_percentile_plot(sub_ax[0],n_idx.keys(),LT_klauAccuracy[0,:,:],LT_Klau_color)
    make_percentile_plot(sub_ax[0],n_idx.keys(),LT_TabuAccuracy[0,:,:],LT_Tabu_color)
    make_percentile_plot(sub_ax[1],n_idx.keys(),triMatch[0,:,:],LT_color)
    make_percentile_plot(sub_ax[1],n_idx.keys(),LT_klauTriMatch[0,:,:],LT_Klau_color)
    make_percentile_plot(sub_ax[1],n_idx.keys(),LT_TabuTriMatch[0,:,:],LT_Tabu_color)
    """
    n_exps = [
        (accuracy,triMatch,LT_color,LT_linestyle),
        (LRT_accuracy,LRT_triMatch,LRT_color,LRT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRM_lrm_Klau_color),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_linestyle),
    ]

    for (acc,tri,c,linestyle) in n_exps:
        make_percentile_plot(sub_ax[0],n_idx.keys(),acc[0,:,:],c,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],n_idx.keys(),tri[0,:,:],c,linestyle=linestyle)


    for ax in sub_ax:
        ax.set_xlim(min(n_idx.keys()),max(n_idx.keys()))
        ax.set_xticks([100, 250, 500, 1000, 1250, 1500])
        ax.set_xlim(75,1525)

    sub_ax[1].set_xticklabels(["100",r"${\bf 250}$","500","1000","1250","1500"],rotation=60)
    sub_ax[1].set_xlabel(r"$|V_A|$")
    
    axes[1,1].annotate(r"$\Lambda$T-"+"Klau",xy=(.3,.75),color=LT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-26)
    axes[1,1].annotate(r"$\Lambda$-TAME",xy=(.4,.25),color=LT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-15)
    axes[1,1].annotate("LR-TAME",xy=(.22,.1),color=LRT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-20)
    axes[1,1].annotate("LRT-Klau",xy=(.625,.425),color=LRT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-7.5)
    axes[1,1].annotate(r"$\Lambda$T-"+"Tabu",xy=(.775,.675),color=LT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-10)
    axes[1,1].annotate("LRT-Tabu",xy=(.775,.875),color=LRT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-5)

    axes[0,0].set_ylabel("accuracy")
    #axes[1,0].set_ylabel(r"matched tris$/ \min{\{|T_A|,|T_B|\}}$")
    

    for ax in axes[0,:]:
        ax.set_xticklabels([])
    #for ax in axes[:,1]:
    #    ax.set_yticklabels([])

    for ax in axes.reshape(-1):
        ax.set_ylim(-.05,1.05)
        ax.grid(True)

    #plt.tight_layout()
    #fig.canvas.draw()
    #fraction_ylabel(fig,axes[1,0])

    axes[0,0].set_ylabel("accuracy")
    axes[1,1].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=5)
    xloc = 1.18
    ax.annotate('', xy=(xloc, .2), xycoords='axes fraction', xytext=(xloc, 0.8),
                arrowprops=dict(arrowstyle="-", color='k'))
    for ax in axes[0,:]:
        ax.set_xticklabels([])

    for ax in axes[1,:]:
        ax.tick_params(axis="x",direction="out",pad=1)

    axes[0,0].set_yticklabels([])
    axes[1,1].set_yticklabels([])
    axes[1,1].yaxis.set_label_position("right")


    for ax in axes[:,1]:
        ax.yaxis.tick_right()

    for ax in axes.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_ylim(-.05,1.05)
        ax.grid(True)


    checkboard_color = [.925]*3
    parity = 1
    for i in range(n):
        for j in range(m):  
            if parity == 1:
                axes[i,j].patch.set_facecolor(checkboard_color)
                #all_gs_ax[i,j].patch.set_alpha(1.1)
        parity *= -1



    plt.show()

"""
def test_patch():
    f = plt.figure()
    ax = plt.gca()


    text = ax.annotate("test",xy=(.5,.5))

    tb = text.get_tightbbox(f.canvas.get_renderer()).transformed(f.transFigure.inverted())
    print(text.get_window_extent(f.canvas.get_renderer()).transformed(f.transFigure.inverted()))
    print(tb)
    #tb = text.get_tightbbox(f.canvas.get_renderer()).transformed(ax.transFigure())
    print(tb.width)
    print((tb.x1 - tb.x0))
    #for x in sorted(dir(tb)):
    #    print(x)
    ax.annotate('', xy=(tb.xmin,tb.y0), xytext=(tb.xmax,tb.y0),xycoords="figure fraction",
                                                arrowprops=dict(arrowstyle="-", color='k'))

    (x0,y0) = ax.transData.transform((tb.x0,tb.y0))
    (x1,y1) = ax.transData.transform((tb.x1,tb.y1))
    

    r = patches.Rectangle((tb.x0,tb.y0),facecolor="none",edgecolor="k",
                           width =(tb.x1 - tb.x0), height=(tb.y1-tb.y0))
    r2 = patches.Rectangle((x0,y0),facecolor="none",edgecolor="k",
                           width =(x1 - x0), height=(y1-y0))

    #ax.add_patch(r)
    #ax.add_patch(r2)
    print(tb)
    plt.show()
    """

def RandomGeometricRG_PostProcessing_is_needed():

    data_location = TAME_RESULTS + "klauExps/"
    fig = plt.figure(figsize=(4.5,4.5))

    #fig, axes = plt.subplots(2,2,figsize=(4.25,5))
    n = 2
    m = 2
    spec = fig.add_gridspec(nrows=n, ncols=m,hspace=0.1,wspace=0.15,left=.1,right=.85,top=.95,bottom=.15)
    all_ax = []
    axes = np.empty((n,m),object)
    for i in range(n):
        for j in range(m):
            ax = fig.add_subplot(spec[i,j])
            axes[i,j] = ax



    def process_ER_Noise_data(data,version="klau_old"):
        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {p:i for (i,p) in enumerate(sorted(set([datum[2] for datum in data])))}
        trials = int(len(data)/(len(p_idx)*len(n_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauAccuracy = np.zeros((len(p_idx),len(n_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx)),int)


        for datum in data:

            if version == "klau_new":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fbounds) = datum
            elif version == "klau_old":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "tabu":
                (seed,p,n,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            else:
                raise ValueError("only supports ")
            i = p_idx[p]
            j = n_idx[n]

            accuracy[i,j,trial_idx[i,j]] = acc
            LT_klauAccuracy[i,j,trial_idx[i,j]] = klau_acc
            triMatch[i,j,trial_idx[i,j]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,trial_idx[i,j]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j] += 1
        
        return accuracy, LT_klauAccuracy, triMatch, LT_klauTriMatch, p_idx, n_idx 

    def process_Dup_Noise_data(data,version="old"):

        p_idx = {p:i for (i,p) in enumerate(sorted(set([datum[1] for datum in data])))}
        n_idx = {n:i for (i,n) in enumerate(sorted(set([datum[2] for datum in data])))}
        sp_idx = {sp:i for (i,sp) in enumerate(sorted(set([datum[3] for datum in data])))}

        trials = int(len(data)/(len(p_idx)*len(n_idx)*len(sp_idx)))

        accuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        klauAccuracy = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        triMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        LT_klauTriMatch = np.zeros((len(p_idx),len(n_idx),len(sp_idx),trials))
        trial_idx = np.zeros((len(p_idx),len(n_idx),len(sp_idx)),int)
        print(f"p vals: {p_idx}")
        print(f"n vals: {n_idx}")
        print(f"sp vals: {sp_idx}")


        for datum in data:
            
            if version == "old":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt) = datum
            elif version == "new klau":
                (seed,p,n,sp,acc,dup_tol_acc,matched_tris,tri_A,tri_B,_,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt,L_sparsity,fstatus) = datum
            else:
                print(f"only supports 'old' and 'new klau', got {version}")

            i = p_idx[p]
            j = n_idx[n]
            k = sp_idx[sp]
            accuracy[i,j,k,trial_idx[i,j,k]] = acc
            klauAccuracy[i,j,k,trial_idx[i,j,k]] = klau_acc
            triMatch[i,j,k,trial_idx[i,j,k]] = matched_tris/min(tri_A,tri_B)
            LT_klauTriMatch[i,j,k,trial_idx[i,j,k]] = klau_tri_match/min(tri_A,tri_B)
            trial_idx[i,j,k] += 1
        
        return accuracy, klauAccuracy, triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx  


    def make_percentile_plot(plot_ax, x_domain,data,color,hatch=None,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.05,color)
        ]
        
        #plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)
        n,m = data.shape
        percentile_linewidth=.01

        for (lower_percentile,upper_percentile,alpha,color) in ribbons:
            #plot_ax.plot(np.percentile(data.T, lower_percentile, axis=0),c=color,linewidth=percentile_linewidth)
            #plot_ax.plot(np.percentile(data.T, upper_percentile, axis=0),c=color,linewidth=percentile_linewidth)

            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor=color,alpha=.1,edgecolor=color)
            """
            plot_ax.fill_between(x_domain,
                            np.percentile(data.T, lower_percentile, axis=0),
                            np.percentile(data.T, upper_percentile, axis=0),
                            facecolor="None",hatch=hatch,edgecolor=color,alpha=.4)
            """

        for (col_func,alpha,color) in lines:
            line_data = []

            for i in range(n):
                line_data.append(col_func(data[i,:]))
        
            plot_ax.plot(x_domain,line_data,alpha=alpha,c=color,**kwargs)

    #
    #  Erdos Reyni Noise 
    #

    #file="LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.01]_postProcess:KlauAlgo_trialcount:20.json"
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json'
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    #file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.01]_postProcess:TabuSearch_trialcount:20.json'
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json'
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy, _, LT_TabuTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"tabu")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_KlauAccuracy, LRT_triMatch, LRT_KlauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _, LRT_TabuAccuracy,_,LRT_TabuTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"tabu")


    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy, _, LRT_lrm_KlauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy, LREA_TriMatch, LREA_KlauTriMatch, p_idx, n_idx = process_ER_Noise_data(json.load(f),"klau_new")

    file = "LowRankEigenAlign_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:ErdosReyni_p:[0.05]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy, _, LREA_TabuTriMatch, _, _ = process_ER_Noise_data(json.load(f),"tabu")





    """
    make_percentile_plot(sub_ax[0],n_idx.keys(),accuracy[0,:,:],LT_color)
    make_percentile_plot(sub_ax[0],n_idx.keys(),LT_klauAccuracy[0,:,:],LT_Klau_color)
    make_percentile_plot(sub_ax[0],n_idx.keys(),LT_TabuAccuracy[0,:,:],LT_Tabu_color)
    make_percentile_plot(sub_ax[1],n_idx.keys(),triMatch[0,:,:],LT_color)
    make_percentile_plot(sub_ax[1],n_idx.keys(),LT_klauTriMatch[0,:,:],LT_Klau_color)
    make_percentile_plot(sub_ax[1],n_idx.keys(),LT_TabuTriMatch[0,:,:],LT_Tabu_color)
    """

    sub_ax = axes[:,0]
    spectral_embedding_exps = [
        (accuracy,triMatch,LT_color,LT_linestyle),
        (LRT_accuracy,LRT_triMatch,LRT_color,LRT_linestyle),
        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,LREigenAlign_linestyle),
    ]


    post_processing_exps = [
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,LT_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRM_lrm_Klau_color),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,LRT_Klau_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,LRT_Tabu_linestyle),
        (LREA_TabuAccuracy, LREA_TabuTriMatch,LREA_Tabu_color,LREA_Tabu_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,LREA_Klau_linestyle),
    ]

    for (acc,tri,c,linestyle) in chain(spectral_embedding_exps,post_processing_exps):
        make_percentile_plot(axes[0,0],n_idx.keys(),acc[0,:,:],c,linestyle=linestyle)
        make_percentile_plot(axes[1,0],n_idx.keys(),tri[0,:,:],c,linestyle=linestyle)
    """
    for (acc,tri,c,linestyle) in post_processing_exps:
        make_percentile_plot(axes[0,1],n_idx.keys(),acc[0,:,:],c,linestyle=linestyle)
        make_percentile_plot(axes[1,1],n_idx.keys(),tri[0,:,:],c,linestyle=linestyle)
    """

    
    #axes[0,0].set_ylabel("accuracy")
    #axes[1,0].set_ylabel(r"matched tris$/ \min{\{|T_A|,|T_B|\}}$")
    
    #
    #  Duplcation Noise 
    #
    default_p = .5
    default_sp = .25

    sub_ax = axes[:,1]
    """
    sub_ax = [fig.add_subplot(spec[i,1]) for i in [0,1]]
    all_ax.append(sub_ax)
    all_ax_gs[:,1] = sub_ax
    """
    #sub_ax = axes[:,1]
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.75]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    file = "LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        accuracy,  LT_klauAccuracy,triMatch,LT_klauTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f))
    
    file = 'LambdaTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json' 
    with open(data_location + file,'r') as f:
        _,  LT_TabuAccuracy,_,LT_TabuTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LRT_accuracy,  LRT_TabuAccuracy,LRT_triMatch,LRT_TabuTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f))
    
    file = "LowRankTAME_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_KlauAccuracy,_,LRT_KlauTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f),"new klau")
    
    file = "LowRankTAME-lrm_graphType:RG_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LRT_lrm_KlauAccuracy,_,LRT_lrm_KlauTriMatch, p_idx, n_idx, sp_idx = process_Dup_Noise_data(json.load(f),"new klau")
    
    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:KlauAlgo_trialcount:20.json"
    with open(data_location + file,'r') as f:
        LREA_Accuracy,  LREA_KlauAccuracy,LREA_TriMatch,LREA_KlauTriMatch, _, _, _ = process_Dup_Noise_data(json.load(f),"new klau")


    file ="LowRankEigenAlign_graphType:RG_degdist:LogNormal-log5_n:[100,250,500,1000,1250,1500]_noiseModel:Duplication_p:[0.5]_sp:[0.25]_postProcess:TabuSearch_trialcount:20.json"
    with open(data_location + file,'r') as f:
        _,  LREA_TabuAccuracy,_,LREA_TabuTriMatch, _, _, _ = process_Dup_Noise_data(json.load(f))

    n_exps = [
        (accuracy,triMatch,LT_color,None,LT_linestyle),
        (LT_klauAccuracy,LT_klauTriMatch,LT_Klau_color,None,LT_Klau_linestyle),
        (LT_TabuAccuracy,LT_TabuTriMatch,LT_Tabu_color,None,LT_Tabu_linestyle),
        
        (LRT_accuracy,LRT_triMatch,LRT_color,None,LRT_linestyle),
        (LRT_TabuAccuracy,LRT_TabuTriMatch,LRT_Tabu_color,None,LRT_Tabu_linestyle),
        (LRT_KlauAccuracy,LRT_KlauTriMatch,LRT_Klau_color,None,LRT_Klau_linestyle),

        (LREA_Accuracy, LREA_TriMatch,LREigenAlign_color,None,LREigenAlign_linestyle),
        (LREA_KlauAccuracy, LREA_KlauTriMatch,LREA_Klau_color,None,LREA_Klau_linestyle),
        (LREA_TabuAccuracy,LREA_TabuTriMatch,LREA_Tabu_color,None,LREA_Tabu_linestyle),
        #(LRT_lrm_KlauAccuracy,LRT_lrm_KlauTriMatch,LRT_lrm_Klau_color),
    ]

    for (acc, tri, c,hatch,linestyle) in n_exps:
        make_percentile_plot(sub_ax[0],n_idx.keys(),acc[p_idx[default_p],:,sp_idx[default_sp],:],c,linestyle=linestyle)
        make_percentile_plot(sub_ax[1],n_idx.keys(),tri[p_idx[default_p],:,sp_idx[default_sp],:],c,linestyle=linestyle)

    
    #sub_ax[1].set_xticklabels(["100",r"${\bf 250}$","500","1000","1250","1500"],rotation=60)
    #sub_ax[1].set_xlabel(r"$|V_A|$")

    #
    #   Final touches on axis
    #
    title_size = 12
    #"Erdős Rényi"+"\nNoise"
    #bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=.05)
    bbox = dict(boxstyle="round", fc="w",ec="w",alpha=1.0,pad=.05)
    axes[0,0].annotate(u"Erdős Rényi",xy=(.975,.975), xycoords='axes fraction',ha="right",va="top",fontsize=title_size).set_bbox(bbox)
    axes[0,1].annotate("Duplication",xy=(.975,.975), xycoords='axes fraction',ha="right",va="top",fontsize=title_size).set_bbox(bbox)


    axes[1,1].annotate(r"$\Lambda$-TAME",xy=(.4,.25),color=LT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-15)
    axes[1,1].annotate("LR-TAME",xy=(.4,.11),color=LRT_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-10)
    axes[1,1].annotate("LREA",xy=(.01,.005),color=LREigenAlign_color,xycoords="axes fraction",ha="left",va="bottom",fontsize=10,rotation=0)


    axes[1,0].annotate("LRT-Klau",xy=(.575,.325),color=LRT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-5)
    axes[1,0].annotate(r"$\Lambda$T-"+"Klau",xy=(.6,.62),color=LT_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-26)
    axes[1,0].annotate("LREA-Klau",xy=(.275,.55),color=LREA_Klau_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-35)
    

    axes[0,1].annotate(r"$\Lambda$T-"+"Tabu",xy=(.775,.525),color=LT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-2.5)
    axes[0,1].annotate("LRT-Tabu",xy=(.775,.74),color=LRT_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=0)
    axes[0,1].annotate("LREA-Tabu",xy=(.375,.575),color=LREA_Tabu_color,xycoords="axes fraction",ha="center",va="center",fontsize=10,rotation=-40)



    for ax in axes[0,:]:
        ax.set_xticklabels([])

    for ax in axes[1,:]:
        ax.tick_params(axis="x",direction="out",pad=1)
        ax.set_xticklabels(["100",r"${\bf 250}$","500","1000","1250","1500"],rotation=60)
        ax.set_xlabel(r"$|V_A|$")

    for ax in axes.reshape(-1):
        ax.set_ylim(0.0,1.00)
        ax.grid(True)

    #plt.tight_layout()
    #fig.canvas.draw()
    #fraction_ylabel(fig,axes[1,0])

    axes[0,0].set_ylabel("accuracy")
    axes[1,1].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$",labelpad=5)
    xloc = 1.16
    ax.annotate('', xy=(xloc, .2), xycoords='axes fraction', xytext=(xloc, 0.8),
                arrowprops=dict(arrowstyle="-", color='k'))

    axes[0,0].set_yticklabels([])
    axes[1,1].set_yticklabels([])
    axes[1,1].yaxis.set_label_position("right")


    for ax in axes[:,1]:
        ax.yaxis.tick_right()

    for ax in axes.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)
        ax.set_ylim(-.00,1.00)
        ax.grid(True)


        ax.set_xlim(min(n_idx.keys()),max(n_idx.keys()))
        ax.set_xticks([100, 250, 500, 1000, 1250, 1500])
        ax.set_xlim(100,1500)


    """
    #checkboard_color = [.925]*3
    parity = 1
    for i in range(n):
        for j in range(m):  
            if parity == 1:
                axes[i,j].patch.set_facecolor(checkboard_color)
                #all_gs_ax[i,j].patch.set_alpha(1.1)
        parity *= -1
    """


    plt.show()



#
#   K nearest Neighbors Post Processing Experiments
#
def klauKNearestRedo():

    plt.rc('text.latex', preamble=r'\usepackage{/Users/charlie/Documents/Code/TKPExperimentPlots/latex/dgleich-math}')
    
    fig = plt.figure(figsize=(6,3.5))
    global_ax = plt.gca()


    data_location = TAME_RESULTS + "klauExps/"
    def process_klau_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        runtime = np.zeros((len(order_idx),len(k_idx),trials))
        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        postProcessingRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}
        A_motifs_counts = {}


        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]
            print(f"p:{results[0][1]} n:{results[0][2]} sp:{results[0][3]}")
    
            accuracy[i,j,:] = [x[4] for x in results]
            if version=="LT":
                runtime[i,j,:] = [sum([sum(val) for val in result[11].values()]) for result in results] 
                postProcessingAccuracy[i,j,:] = [x[15] for x in results]
                postProcessingRuntime[i,j,:] = [x[17] + x[18] for x in results]
                     #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 
                    vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                    A_motifs_counts[order]= [x[7][0] for x in results]

            elif version=="LRT":
                postProcessingAccuracy[i,j,:] = [x[14] for x in results]
                postProcessingRuntime[i,j,:] = [x[16] + x[17] for x in results]

        if version=="LT":
            return accuracy,runtime, postProcessingAccuracy, postProcessingRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
        else:
            return accuracy, postProcessingAccuracy, postProcessingRuntime,order_idx, k_idx #triMatch,LT_klauTriMatch, p_idx, n_idx 
    
    def process_tabu_data(data,version="LT"):
        order_idx = {order:i for (i,order) in enumerate(sorted(set([datum[0] for datum in data])))}
        k_idx = {k:i for (i,k) in enumerate(sorted(set([datum[1] for datum in data])))}
        trials =len(data[0][-1])

        accuracy = np.zeros((len(order_idx),len(k_idx),trials))
        tabuAccuracy = np.zeros((len(order_idx),len(k_idx),trials))
        triMatch = np.zeros((len(order_idx),len(k_idx),trials))
        tabuRuntime = np.zeros((len(order_idx),len(k_idx),trials))
        #sparsity = np.zeros((len(order_idx),len(k_idx),trials))
        trial_idx = np.zeros((len(order_idx),len(k_idx)),int)

        vertex_coverage = {}



        for (order,k,results) in data:
            # expecting results:
            #     (seed,p, n, sp, acc, dup_tol_acc, matched_matching_score, A_motifs, B_motifs, A_motifDistribution, B_motifsDistribution,profiling,edges,klau_edges,klau_tri_match,klau_acc,_,klau_setup,klau_rt)
            i = order_idx[order]
            j = k_idx[k]
            accuracy[i,j,:] = [x[4] for x in results]
            if version == "LT":
                tabuAccuracy[i,j,:] = [x[15] for x in results]
                tabuRuntime[i,j,:] = [x[18] for x in results]
                #sparsity[i,j,:] = [x[19] for x in results]
                if k == min(k_idx.keys()): #only do this once over k 
                    vertex_coverage[order] = [len(list(filter(lambda y: y!= 0.0,x[9][0])))/x[2] for x in results]
                    

            elif version == "LRT":
                print(i," ",j)
                tabuAccuracy[i,j,:] = [x[14] for x in results]
                tabuRuntime[i,j,:] = [x[17] for x in results]

        if version == "LT":
            return accuracy, tabuAccuracy,tabuRuntime,vertex_coverage, order_idx, k_idx
        else:
            return accuracy, tabuAccuracy,tabuRuntime,order_idx, k_idx
    

    #filename = "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    filename = "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_KAmiter:1000_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    #filename = "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    with open(data_location+filename,'r') as f:
 #   with open(data_location+filename,"r") as f:
        #return json.load(f)
        accuracy,LT_runtime, LT_klauAccuracy,LT_klauRuntime,vertex_coverage,A_motifs_counts, order_idx, k_idx = process_klau_data(json.load(f))

    filename = "RandomGeometric_degreedist:LogNormal_TabuSearchkvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        _, LT_TabuAccuracy, LT_TabuRuntime, _, order_idx, k_idx = process_tabu_data(json.load(f))


    filename = "RandomGeometric_LRTAME_degreedist:LogNormal_KlauAlgokvals:[15,30,45,60,75,90]_n:[500]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        _, LRT_klauAccuracy, LRT_klauRuntime, order_idx, k_idx = process_klau_data(json.load(f),"LRT")


    filename = "RandomGeometric_LRTAME_degreedist:LogNormal_TabuSearchkvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json"
    with open(data_location+filename,'r') as f:
        _, LRT_TabuAccuracy, LRT_TabuRuntime, order_idx, k_idx = process_tabu_data(json.load(f),"LRT")
    


    def make_percentile_plot(plot_ax, x_domain,data,color,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (20,80,.2,color)
        ]
        plot_percentiles(plot_ax,  data.T, x_domain, lines, ribbons,**kwargs)

    def make_violin_plot(ax,data,precision=2,c=None,v_alpha=.8):
        v = ax.violinplot(data, points=100, positions=[.5], showmeans=False, 
                       showextrema=False, showmedians=True,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y0-.1),(x0,y0 + (y1-y0)/2 -.1)],[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate(f"{np.median(data):.{precision}f}",xy=(.5,.38),xycoords="axes fraction",ha="center",fontsize=10)
        ax.annotate(f"{np.min(data):.{precision}f}",xy=(0,.8),xycoords="axes fraction",ha="left",fontsize=6,alpha=.8)
        ax.annotate(f"{np.max(data):.{precision}f}",xy=(1,.1),xycoords="axes fraction",ha="right",fontsize=6,alpha=.8)

        if c is not None:
            v["cmedians"].set_color("k")
            v["cmedians"].set_alpha(.3)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)            
                b.set_alpha(v_alpha)
                #b.set_alpha(1)
                b.set_color(c)



    def make_violin_plot_legend(ax,c="k"):
        
        v = ax.violinplot([np.random.normal() for i in range(50)], points=100, positions=[.5], showmeans=False, 
                        showextrema=False, showmedians=True,widths=.5,vert=False)#,quantiles=[[.2,.8]]*len(data2))
        #ax.axes.set_axis_off()

        #  --  update median lines to have a gap  --  #
        ((x0,y0),(_,y1)) = v["cmedians"].get_segments()[0]
        newMedianLines = [[(x0,y0-.1),(x0,y0 + (y1-y0)/2 -.1)],[(x0,y0 + (y1-y0)/2 +.1),(x0,y1+.1)]]
        v["cmedians"].set_segments(newMedianLines)

        ax.annotate("med.",xy=(.5,.5),xycoords="axes fraction",ha="center",va="center",fontsize=10)
        ax.annotate(f"min",xy=(0,.8),xycoords="axes fraction",ha="left",fontsize=6,alpha=.8)
        ax.annotate(f"max",xy=(1,.1),xycoords="axes fraction",ha="right",fontsize=6,alpha=.8)

        if c is not None:
            v["cmedians"].set_color(c)
            for b in v['bodies']:
                b.set_facecolor(c)
                b.set_edgecolor(c)
                b.set_alpha(.2)
                b.set_color(c)

    #  -- add in order labels overhead --  #
    #global_ax.set_xlim(0,1)
    #pad = 1.0/(2*len(order_idx))
    #global_ax.set_xticks(np.linspace(pad,1-pad,len(order_idx)))
    #global_ax.set_xticklabels(order_idx.keys())
    global_ax.set_yticklabels([])
    global_ax.set_yticklabels([])
    #global_ax.yaxis.set_ticks_position('none')

    #global_ax.xaxis.set_ticks_position("top")
    #global_ax.xaxis.set_label_position("top")
    #global_ax.xaxis.set_label_coords(-.06, 1.0)
    #global_ax.set_xlabel("Clique\n Size",ha="center")

    widths = [.5, 3, 2, 1,1]
    #heights = [1]*len(order_idx)
    spec = fig.add_gridspec(nrows=5,ncols=1+len(order_idx),hspace=0.0,wspace=0.0,height_ratios=widths,left=.15,right=.95)

    allCAx = []
    allAccAx = [] 
    allRtAx = [] 
    allVCAx = []
    allMotifCountAx = []
    allSparsityAx = []
    first = True
    annotation_idx =1
    if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
        k_tick_idx = 0
    else:
        k_tick_idx = 3
    parity = 0
    for idx,(order,i) in enumerate(order_idx.items()):
        #
        #  Clique size
        #
        ax = fig.add_subplot(spec[0,i])
        
        allCAx.append(ax)
        if idx % 2 == parity:
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        ax.annotate(f"{order}",xy=(.5, .5), xycoords='axes fraction', c="k",size=10,ha="center",va="center")

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        #
        #  Accuracy Plots
        #
        if idx == annotation_idx:
            ax = fig.add_subplot(spec[1,i],zorder=10)#,sharey=allAccAx[0])
        else:
            ax = fig.add_subplot(spec[1,i])#,sharey=allAccAx[0])
        allAccAx.append(ax)

        if idx % 2 != parity:
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        
        ax.set_yticks([0.0,0.25,.5,.75,1.0])
        ax.set_ylim(0,1.0)

        #ax.set_xticks([])

        
        
        plt.axhline(y=np.median(accuracy[i,:,:]), color=LT_color,linestyle=LT_linestyle)
        plt.axhline(y=np.max(accuracy[i,:,:]), color=LT_color,linestyle="dotted")
        make_percentile_plot(ax,k_idx.keys(),LT_klauAccuracy[i,:,:],LT_Klau_color,linestyle=LT_Klau_linestyle)
        make_percentile_plot(ax,k_idx.keys(),LT_TabuAccuracy[i,:,:],LT_Tabu_color,linestyle=LT_Tabu_linestyle)

        #make_percentile_plot(ax,k_idx.keys(),LRT_klauAccuracy[i,:,:],LRT_Klau_color)
        #make_percentile_plot(ax,k_idx.keys(),LRT_TabuAccuracy[i,:,:],LRT_Tabu_color)

        ax.set_xticklabels([])
        ax.set_yticklabels([])


        #
        #  Runtime Plots
        #
        
        if idx == k_tick_idx:
            ax = fig.add_subplot(spec[2,i],zorder=3)
        else:
            ax = fig.add_subplot(spec[2,i])
        allRtAx.append(ax)
        
        if idx % 2 == parity:
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)

        if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),600)
        elif filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),1200)
        else:
            ax.set_ylim(np.min(LT_klauRuntime[:,:,:]),2100)
        #ax.set_yscale("log")
        if idx != 0:
            ax.set_yticklabels([])
        ax.set_xticks([15,45,90])

        ax.set_xticklabels([])
            
        make_percentile_plot(ax,k_idx.keys(),LT_klauRuntime[i,:,:],LT_Klau_color,linestyle=LT_Klau_linestyle)
        make_percentile_plot(ax,k_idx.keys(),LT_TabuRuntime[i,:,:],LT_Tabu_color,linestyle=LT_Tabu_linestyle)
        #make_percentile_plot(ax,k_idx.keys(),LT_runtime[i,:,:],LT_color,linestyle=LT_linestyle)
        plt.axhline(y=np.median(LT_runtime[i,:,:]), color=LT_color,linestyle=LT_linestyle)
        
        #make_percentile_plot(ax,k_idx.keys(),LRT_klauRuntime[i,:,:],LRT_Klau_color)
        #make_percentile_plot(ax,k_idx.keys(),LRT_TabuRuntime[i,:,:],LRT_Tabu_color)

        """
        #
        # Sparsity
        #
        ax = fig.add_subplot(spec[3,i])
        allSparsityAx.append(ax)
        
        if idx % 2 != parity:
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)

        ax.set_ylim(0.01,.3)
        ax.set_yticks([.1,.2,.3])
        ax.set_xticks([15,45,90])


        ax.set_xticklabels([])
    
        make_percentile_plot(ax,k_idx.keys(),sparsity[i,:,:],t3_color)
        """

        #
        #  Vertex Coverage
        #
        ax = fig.add_subplot(spec[3,i])
        
        allVCAx.append(ax)

        
        if idx % 2 != parity:
            make_violin_plot(ax,vertex_coverage[order],c="w")
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        else:
            make_violin_plot(ax,vertex_coverage[order],c="k",v_alpha=.1)


        #
        #  Motif Counts
        #
        
        ax = fig.add_subplot(spec[4,i])
        
        allMotifCountAx.append(ax)

        if idx % 2 == parity:
            make_violin_plot(ax,A_motifs_counts[order],c="w",precision=0)
            ax.patch.set_facecolor('k')
            ax.patch.set_alpha(0.1)
        else:
            make_violin_plot(ax,A_motifs_counts[order],c="k",v_alpha=.1,precision=0)



    violinLegendAx = fig.add_subplot(spec[3,-1])

    for ax in chain(allAccAx,allRtAx,allCAx,allVCAx,allMotifCountAx,allSparsityAx,[global_ax],[violinLegendAx]):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in chain(allAccAx,allRtAx,allSparsityAx):
        ax.grid(True)
        ax.set_xticks([15,45,90])
        ax.set_xlim(min(k_idx.keys()),max(k_idx.keys()))

    #  - make violin plot legend -  #
    make_violin_plot_legend(violinLegendAx)

    #  -- Add in x-domain annotations mid plot --  #

    allRtAx[k_tick_idx].xaxis.set_label_position("top")
    allRtAx[k_tick_idx].xaxis.set_ticks_position('top')
    allRtAx[k_tick_idx].tick_params(axis="x",direction="out", pad=-15)
    allRtAx[k_tick_idx].set_xticklabels([15,45,90],zorder=5)
    #allRtAx[3].set_axisbelow(False)
    #allRtAx[0].set_xlabel("nearest\nneighbors")
    allRtAx[k_tick_idx].annotate("nearest neighbors",xy=(.5,1.1),ha="center",xycoords='axes fraction')
    
    #  -- Alternate tick labels to opposite axes --  #

    #allCAx[0].yaxis.set_label_coords(-.05, .5)
    allCAx[0].annotate("Clique Size",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')
    #allCAx[0].set_ylabel("Clique\nSize",rotation=0,labelpad=-2,loc="center")
    
    #allAccAx[-1].yaxis.set_label_position("right")
    allAccAx[-1].yaxis.set_ticks_position('right')
    allAccAx[-1].set_yticklabels([0,.25,.5,.75,1.0])
    allAccAx[-1].tick_params(axis="both",direction="out",which='both', length=7.5)
    allAccAx[0].set_ylabel("Accuracy",rotation=0,labelpad=0,ha="right")
    
    allRtAx[-1].yaxis.set_label_position("right")
    #allRtAx[-1].set_ylabel("Klau \nruntime\n (seconds)",rotation=0,loc="center")
    #allRtAx[-1].annotate("Post\nProcessing\nRuntime\n",xy=(1.1,.3),ha="left",va="center",xycoords='axes fraction')
    #allRtAx[0].yaxis.set_ticks_position('left')
    allRtAx[0].annotate("Runtime (s)",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')
    allRtAx[-1].yaxis.set_ticks_position('right')
    if filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
        allRtAx[0].set_yticklabels(["10 s","2 min","5 min","10 min"])
        for ax in allRtAx:
            ax.set_yticks([1e0,1e1,120,300,600])   
    elif filename == "RandomGeometric_degreedist:LogNormal_kvals:[15,30,45,60,75,90]_noiseModel:Duplication_n:[500]_orders:[2,3,4,5,6,7,8,9]_samples:1000000_trials:25_data.json":
        allRtAx[0].set_yticklabels(["10 s","5 min","10 min","15 min","20 min"])
        for ax in allRtAx:
            ax.set_yticks([1e1,300,600,900,1200])
    else:
        #allRtAx[0].set_yticklabels(["10 s","10 min","20 min","35 min"])
        for i,ax in enumerate(allRtAx):
            
            ax.set_yscale("log")
            #ax.set_yticks([1e1,600,1200,2100])
            ax.set_yticks([1e0,1e1,1e2,1e3,1e4])
            ax.set_ylim(1e-1,2e4)
            if i == len(allRtAx)-1:
                ax.set_yticklabels([r"$10^0$",r"$10^1$",r"$10^2$",r"$10^3$",None])
            else:
                ax.set_yticklabels([])

    """ old sparsity axes labeling
    allSparsityAx[0].annotate("L Sparsity\n",xy=(-.1,.3),ha="right",va="center",xycoords='axes fraction')
    allSparsityAx[-1].yaxis.set_ticks_position('right')
    allSparsityAx[-1].set_yticklabels([.1,.2,.3])
    """


    allVCAx[0].annotate("Vertex\nCoverage",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')
    #  -- Annotate accuracy plots --  #
    #allAccAx[2].annotate("Klau Post\n Processing", xy=(.5, .75), xycoords='axes fraction', c=t3_color,size=8,ha="center")
    allAccAx[5].annotate(r"$\Lambda$T"+"-Klau", xy=(.5, .675), xycoords='axes fraction', c=LT_Klau_color,size=10,ha="center",rotation=20)
    allAccAx[5].annotate(r"$\Lambda$T"+"-Tabu", xy=(.475, .4), xycoords='axes fraction', c=LT_Tabu_color,size=10,ha="center")
    allAccAx[annotation_idx].annotate("maximum "+r"$\Lambda$"+"-TAME", xy=(.01, .625), xycoords='axes fraction', c=t2_color,size=10,ha="left")
    allAccAx[annotation_idx].annotate("median "+r"$\Lambda$"+"-TAME", xy=(.01, .4), xycoords='axes fraction', c=t2_color,size=10,ha="left")

    allMotifCountAx[0].annotate("A Motifs",xy=(-.1,.5),ha="right",va="center",xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

def LVGNA_PostProcessing_localizedData(fig = None):

    data_path = TAME_RESULTS + "LVGNA_Experiments/klauPostProcessing/"

    with open(data_path + "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_samples:3000000_data.json","r") as f:
        data = json.load(f)
        vertex_products = []
        edge_products = []
        motif_products = []
        klau_tri_match_ratios = []
        klau_edge_match_ratios = []
        klau_runtimes = []

        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,klau_edges_matched,klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity) in data:
            

            vertex_A = vertex_counts[" ".join(file_i.split(".smat")[0].split("_"))]
            vertex_B = vertex_counts[" ".join(file_j.split(".smat")[0].split("_"))]
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]

            klau_runtimes.append(klau_setup_rt+klau_rt)
            vertex_products.append(vertex_A*vertex_B)
            edge_products.append(edges_A*edges_B)
            motif_products.append(A_Motifs[0]*B_Motifs[0])

            klau_edge_match_ratios.append(klau_edges_matched/min(edges_A,edges_B))
            klau_tri_match_ratios.append(klau_tris_matched/min(A_Motifs[0],B_Motifs[0]))

    #with open(data_path + "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:SuccessiveKlau_samples:3000000_data.json","r") as f:
    with open(data_path + "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:SuccessiveKlau-const-iter:5-maxIter:500_samples:10000000_data.json","r") as f:
        data = json.load(f)
        successive_klau_tri_match_ratios = []
        successive_klau_edge_match_ratios = []
        successive_klau_runtimes = []
     
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,sklau_edges_matched,sklau_tris_matched,_,successive_klau_profiling) in data:
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            successive_klau_edge_match_ratios.append(sklau_edges_matched/min(edges_A,edges_B))
            successive_klau_tri_match_ratios.append(sklau_tris_matched/min(A_Motifs[0],B_Motifs[0]))
            successive_klau_runtimes.append(sum(successive_klau_profiling["runtime"]))

    with open(data_path + "LVGNA_pairwiseAlignment_LambdaTAME_MultiMotif_alphas:[1.0,0.5]_betas:[0.0,1.0,10.0,100.0]_order:3_postProcessing:TabuSearch_samples:30000000_data.json","r") as f:
        data = json.load(f)
        tabu_tri_match_ratios = []
        tabu_edge_match_ratios = []
        tabu_runtimes = []
     
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,A_motifDistribution,B_motifDistribution,LT_profile,LT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,full_rt) in data:
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            tabu_edge_match_ratios.append(tabu_edges_matched/min(edges_A,edges_B))
            tabu_tri_match_ratios.append(tabu_tris_matched/min(A_Motifs[0],B_Motifs[0]))
            tabu_runtimes.append(full_rt)

    def parse_Cpp_data(files):
        # 11/08/2021
        # this TAME C++ data is from the the TAME_tol1e-16/ folder on nilpotent. 


        tri_match_ratio = []
        edge_match_ratio = []
        runtime = []

        Opt_pp_tri_match = np.array([[0.        , 0.31962839, 0.49068984, 0.73699422, 0.38482584,
                                0.56821149, 0.73507463, 0.41034417, 0.21365294, 0.43302679],
                            [0.31962839, 0.        , 0.41226856, 0.56358382, 0.20782911,
                                0.41905515, 0.56343284, 0.19951242, 0.14383607, 0.277489  ],
                            [0.49068984, 0.41226856, 0.        , 0.86849711, 0.59139219,
                                0.        , 0.88246269, 0.52028903, 0.48869203, 0.54138345],
                            [0.73699422, 0.56358382, 0.86849711, 0.        , 0.5216763 ,
                                0.86560694, 0.81529851, 0.59393064, 0.48699422, 0.32514451],
                            [0.38482584, 0.20782911, 0.59139219, 0.5216763 , 0.        ,
                                0.54656424, 0.54104478, 0.2348732 , 0.12322424, 0.26589364],
                            [0.56821149, 0.41905515, 0.        , 0.86560694, 0.54656424,
                                0.        , 0.88432836, 0.59501806, 0.47279833, 0.47141144],
                            [0.73507463, 0.56343284, 0.88246269, 0.81529851, 0.54104478,
                                0.88432836, 0.        , 0.43097015, 0.4738806 , 0.36007463],
                            [0.41034417, 0.19951242, 0.52028903, 0.59393064, 0.2348732 ,
                                0.59501806, 0.43097015, 0.        , 0.24366858, 0.31907237],
                            [0.21365294, 0.14383607, 0.48869203, 0.48699422, 0.12322424,
                                0.47279833, 0.4738806 , 0.24366858, 0.        , 0.21631347],
                            [0.43302679, 0.277489  , 0.54138345, 0.32514451, 0.26589364,
                                0.47141144, 0.36007463, 0.31907237, 0.21631347, 0.        ]])

        Opt_pp_edge_match = np.array([[0.        , 0.03959086, 0.09784676, 0.07616797, 0.09517096,
                    0.09666124, 0.07142857, 0.12066061, 0.05343453, 0.03228293],
                [0.03959086, 0.        , 0.06071178, 0.06407926, 0.07256785,
                    0.05853591, 0.05718136, 0.08418778, 0.05153069, 0.02464035],
                [0.09784676, 0.06071178, 0.        , 0.10525359, 0.13099225,
                    0.        , 0.09819022, 0.26310971, 0.27364573, 0.05234201],
                [0.07616797, 0.06407926, 0.10525359, 0.        , 0.05362661,
                    0.10407199, 0.09020023, 0.06144337, 0.05644428, 0.0428104 ],
                [0.09517096, 0.07256785, 0.13099225, 0.05362661, 0.        ,
                    0.12081424, 0.05429342, 0.07666549, 0.04630772, 0.04436905],
                [0.09666124, 0.05853591, 0.        , 0.10407199, 0.12081424,
                    0.        , 0.09462842, 0.27277721, 0.21094583, 0.03969002],
                [0.07142857, 0.05718136, 0.09819022, 0.09020023, 0.05429342,
                    0.09462842, 0.        , 0.03725452, 0.04745861, 0.04110512],
                [0.12066061, 0.08418778, 0.26310971, 0.06144337, 0.07666549,
                    0.27277721, 0.03725452, 0.        , 0.16306577, 0.04207558],
                [0.05343453, 0.05153069, 0.27364573, 0.05644428, 0.04630772,
                    0.21094583, 0.04745861, 0.16306577, 0.        , 0.02127227],
                [0.03228293, 0.02464035, 0.05234201, 0.0428104 , 0.04436905,
                    0.03969002, 0.04110512, 0.04207558, 0.02127227, 0.        ]])

        post_processing_runtime = np.array([[     0.,  44125.,  81125.,  28571.,  22686., 102315.,  28106.,
                48095.,  67291.,  33496.],
            [ 44125.,      0.,  80297.,  18875.,  14284., 116307.,  19181.,
                41262., 122796.,  24637.],
            [ 81125.,  80297.,      0.,  22153.,  22514.,      0.,  21328.,
                267434., 803348.,  16314.],
            [ 28571.,  18875.,  22153.,      0.,   5217.,   8248.,  48191.,
                10272.,  11771.,   9239.],
            [ 22686.,  14284.,  22514.,   5217.,      0.,  13905.,  41879.,
                10897.,  12437.,   7398.],
            [102315., 116307.,      0.,   8248.,  13905.,      0.,  37933.,
                142752., 622113.,  18830.],
            [ 28106.,  19181.,  21328.,  48191.,  41879.,  37933.,      0.,
                44876.,  41924.,  40548.],
            [ 48095.,  41262., 267434.,  10272.,  10897., 142752.,  44876.,
                    0.,  96492.,   7596.],
            [ 67291., 122796., 803348.,  11771.,  12437., 622113.,  41924.,
                96492.,      0.,  21393.],
            [ 33496.,  24637.,  16314.,   9239.,   7398.,  18830.,  40548.,
                7596.,  21393.,      0.]])


        indexing = {'fly_Y2H1.smat': 9, 'human_PHY2.smat': 8, 'yeast_PHY2.smat': 7, 'human_Y2H1.smat': 1, 'fly_PHY1.smat': 0, 'human_PHY1.smat': 2, 'worm_Y2H1.smat': 6, 'worm_PHY1.smat': 3, 'yeast_PHY1.smat': 5, 'yeast_Y2H1.smat': 4}

        exp_pop = []

        for file_idx,(file_i,file_j) in enumerate(files):
            i = indexing[file_i]
            j = indexing[file_j]



            #having this equal to 0 on a log scale is triggering an error in the svd solvers 
            # data points are cropped
            if post_processing_runtime[i,j] == 0:
                exp_pop.append(file_idx)
                #post_processing_runtime[i,j] = 1e5
            """
            if Opt_pp_tri_match[i,j] == 0.0:
                motif_exp_pop.append(file_idx)
                #pt_pp_tri_match[i,j] = 1.0
            if Opt_pp_edge_match[i,j] == 0.0:
                edge_exp_pop.append(file_idx)
            """


            tri_match_ratio.append(Opt_pp_tri_match[i,j])    
            edge_match_ratio.append(Opt_pp_edge_match[i,j])

            print(f"{file_i} {file_j} runtime: {post_processing_runtime[i,j]}")
            if post_processing_runtime[i,j] > 1e6:
                print(file_i, file_j)
            runtime.append(post_processing_runtime[i,j])

        return tri_match_ratio, edge_match_ratio, runtime, exp_pop

    with open(data_path + "LVGNA_pairwiseAlignment_LowRankEigenAlign_postProcessing:TabuSearch_profile:true_results.json") as f:
        data = json.load(f)
        #for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LREA_profile,LREA_edges_matched,LREA_klau_edges_matched,LREA_klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
        LREA_tabu_tri_match_ratios = []
        LREA_tabu_edge_match_ratios = []
        LREA_tabu_runtimes = []
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LT_profile,LT_edges_matched,tabu_edges_matched,tabu_tris_matched,matching,tabu_profile,full_rt) in data:
            
        

                edges_A = edge_counts[file_i.split(".smat")[0]]
                edges_B = edge_counts[file_j.split(".smat")[0]]
                
                LREA_tabu_edge_match_ratios.append(tabu_edges_matched/min(edges_A,edges_B))
                LREA_tabu_tri_match_ratios.append(tabu_tris_matched/min(A_Motifs,B_Motifs))
                LREA_tabu_runtimes.append(full_rt)

    with open(data_path + "LVGNA_pairwiseAlignment_LowRankEigenAlign_postProcessing:KlauAlgo_profile:true_results.json") as f:
        data = json.load(f)
        
        LREA_klau_tri_match_ratios = []
        LREA_klau_edge_match_ratios = []
        LREA_klau_runtimes = []
    
        for (file_i,file_j,LT_matchedMotifs,A_Motifs,B_Motifs,LREA_profile,LREA_edges_matched,LREA_klau_edges_matched,LREA_klau_tris_matched,_,klau_setup_rt,klau_rt,L_sparsity,f_status) in data:
            edges_A = edge_counts[file_i.split(".smat")[0]]
            edges_B = edge_counts[file_j.split(".smat")[0]]
            
            LREA_klau_edge_match_ratios.append(LREA_klau_edges_matched/min(edges_A,edges_B))
            LREA_klau_tri_match_ratios.append(LREA_klau_tris_matched/min(A_Motifs,B_Motifs))
            LREA_klau_runtimes.append(klau_setup_rt+klau_rt)


    #print(np.min(klau_edge_match_ratios))
    #print(r"$\Lambda$-" +f"TAME edge match res: min - {np.min(klau_edge_match_ratios)} med-{np.median(klau_edge_match_ratios)} max-{np.max(klau_edge_match_ratios)}")
    #print(r"$\Lambda$-" +f"TAME tri match res: min - {np.min(klau_tri_match_ratios)} med-{np.median(klau_tri_match_ratios)} max-{np.max(klau_tri_match_ratios)}")

    tri_match_ratios, edge_match_ratios, runtimes, exp_pop = parse_Cpp_data([(x[0],x[1]) for x in data])
    #print(f"TAME C++ edge match res: min - {np.min(edge_match_ratios)} med-{np.median(edge_match_ratios)} max-{np.max(edge_match_ratios)}")
    #print(f"TAME C++ tri match res: min - {np.min(tri_match_ratios)} med-{np.median(tri_match_ratios)} max-{np.max(tri_match_ratios)}")
    #print(exp_pop)

    #if fig is None:
    fig = plt.figure(figsize=(4.5,6))
    gs = fig.add_gridspec(nrows=3, ncols=2, left=0.1, right=0.95,wspace=0.225,hspace=0.1,top=.97,bottom=.06)

    all_gs_ax = np.empty((3,2),object)

    loess_frac = .3
    
    exps_to_plot = [
        (LREA_klau_edge_match_ratios,LREA_tabu_edge_match_ratios,tabu_edge_match_ratios,successive_klau_edge_match_ratios,klau_edge_match_ratios,edge_match_ratios),
        (LREA_klau_tri_match_ratios,LREA_tabu_tri_match_ratios,tabu_tri_match_ratios,successive_klau_tri_match_ratios,klau_tri_match_ratios,tri_match_ratios),
        (LREA_klau_runtimes,LREA_tabu_runtimes,tabu_runtimes,successive_klau_runtimes,klau_runtimes,runtimes)
    ]
    


    for i,(LREA_klau_exp,LREA_tabu_exp,tabu_exp,successive_klau_exp,klau_exp,tame_exp) in enumerate(exps_to_plot):

        a = 0.0

        #ax = f.add_subplot(gs[i,0])
        #all_gs_ax[i,0] = ax

        cTAME_edge_products = [ep for (i,ep) in enumerate(edge_products) if i not in exp_pop]
        cTAME_motif_products = [mp for (i,mp) in enumerate(motif_products) if i not in exp_pop]
        tame_exp = [x for (i,x) in enumerate(tame_exp) if i not in exp_pop]

        for j,clique_products in enumerate([edge_products,motif_products]):
            print(fig)

            print(gs)
            ax = fig.add_subplot(gs[i,j])
            all_gs_ax[i,j] = ax
            cTAME_clique_products = [cp for (i,cp) in enumerate(clique_products) if i not in exp_pop]
           
            ax.scatter(cTAME_clique_products,tame_exp,c=T_color,zorder=3,alpha=a)
            plot_1d_loess_smoothing(cTAME_clique_products,tame_exp,loess_frac,ax,c=T_color,linestyle=T_linestyle,logFilter=True,logFilterAx="x")
            
            ax.scatter(clique_products,klau_exp,c=LT_Klau_color,zorder=3,alpha=a)
            plot_1d_loess_smoothing(clique_products,klau_exp,loess_frac,ax,c=LT_Klau_color,linestyle=LT_Klau_linestyle,logFilter=True,logFilterAx="both")
            """
            ax.scatter(edge_products,successive_klau_exp,c=t2_color,zorder=3,alpha=a)
            #plot_1d_loess_smoothing(edge_products,successive_klau_exp,loess_frac,ax,c=t2_color,linestyle="dashed",logFilter=True,logFilterAx="both")
            """
            ax.scatter(clique_products,tabu_exp,c=LT_Tabu_color,zorder=3,alpha=a)
            plot_1d_loess_smoothing(clique_products,tabu_exp,loess_frac,ax,c=LT_Tabu_color,linestyle=LT_Tabu_linestyle,logFilter=True,logFilterAx="both")
           
            ax.scatter(clique_products,LREA_tabu_exp,c='r',zorder=3,alpha=a)
            plot_1d_loess_smoothing(clique_products,LREA_tabu_exp,loess_frac,ax,c='r',linestyle=LREigenAlign_linestyle,logFilter=True,logFilterAx="both")
           
            ax.scatter(clique_products,LREA_klau_exp,c='b',zorder=3,alpha=a)
            plot_1d_loess_smoothing(clique_products,LREA_klau_exp,loess_frac,ax,c='b',linestyle=LREigenAlign_linestyle,logFilter=True,logFilterAx="both")
            


        """
        ax.scatter(cTAME_motif_products,tame_exp,c=t1_color,zorder=3,alpha=a)
        plot_1d_loess_smoothing(cTAME_motif_products,tame_exp,loess_frac,ax,c=t1_color,linestyle="solid",logFilter=True,logFilterAx="x")
        
        ax.scatter(motif_products,klau_exp,c=t3_color,zorder=3,alpha=a)
        plot_1d_loess_smoothing(motif_products,klau_exp,loess_frac,ax,c=t3_color,linestyle="dashed",logFilter=True,logFilterAx="both")

        #ax.scatter(motif_products,successive_klau_exp,c=t2_color,zorder=3,alpha=a)
        #plot_1d_loess_smoothing(motif_products,successive_klau_exp,loess_frac,ax,c=t2_color,linestyle="dashed",logFilter=True,logFilterAx="both")

        ax.scatter(motif_products,tabu_exp,c=t4_color,zorder=3,alpha=a)
        plot_1d_loess_smoothing(motif_products,tabu_exp,loess_frac,ax,c=t4_color,linestyle="dashed",logFilter=True,logFilterAx="both")
        """


    for ax in all_gs_ax[2,:]:
        ax.set_yticks([1e2,1e3,1e4,1e5,1e6])
        ax.set_yscale("log")

    for ax in all_gs_ax.reshape(-1):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis="both",direction="out",which='both', length=0)

        #ax.set_xlim(1e6,1e11)
        ax.set_xscale("log")
        #ax.yaxis.set_ticks_position('none')
        #ax.xaxis.set_ticks_position('none')
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])


        ax.grid(True)

    #  make checker board pattern
    n,m = all_gs_ax.shape
    parity = 1
    #checkboard_color = [.925]*3
    """
    for i in range(n):
        for j in range(m):
            if parity == 1:
                all_gs_ax[i,j].patch.set_facecolor(checkboard_color)
                #all_gs_ax[i,j].patch.set_alpha(1.1)
            parity *= -1
        parity *= -1
    """
    # turn off tick labels 
    for i in range(2):
        for ax in all_gs_ax[i,:]:
            ax.set_xticklabels([])


    for i,ylims in enumerate([(0.0001,.35),(0.0001,.95),(1.01e0,3e6)]):
        for ax in all_gs_ax[i,:]:
            ax.set_ylim(*ylims)

    for j,xlims in enumerate([(1e7,2.5e10),(1e5,5e11)]):
        for ax in all_gs_ax[:,j]:
            ax.set_xlim(*xlims)


    for ax in all_gs_ax[:,1]:
        ax.set_yticklabels([])
        ax.set_xticks([1e6,1e8,1e10,1e11])

    for i,ax in enumerate(all_gs_ax[:,0]):
        ax.tick_params(axis="y",which="both",direction="in",pad=5)
        ax.yaxis.set_ticks_position('right')
        ax.set_xticks([1e8,1e9,1e10])
        #bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
        """
        if i % 2 == 0:
            bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=.1)
        else:
            bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
        """

        #for tl in ax.get_yticklabels():
        #    tl.set_bbox(bbox)

    for j,ax in enumerate(all_gs_ax[2,:]):

        ax.set_yticks([1e1,1e2,1e3,1e4,1e5,1e6])
        ax.tick_params(axis="x",which="both",direction="in",pad=-15)
        bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
        """
        if j % 2 == 0:
            bbox = dict(boxstyle="round", ec=checkboard_color, fc=checkboard_color, alpha=1.0,pad=.1)
        else:
            bbox = dict(boxstyle="round", ec="w", fc="w", alpha=1.0,pad=.1)
        """
        for tl in ax.get_xticklabels():
            tl.set_bbox(bbox)


    
    all_gs_ax[1,0].set_yticks([.2,.4,.6,.8])

    all_gs_ax[2,0].set_xlabel(r"$|E_A||E_B|$")
    all_gs_ax[2,1].set_xlabel(r"$|T_A||T_B|$")

    all_gs_ax[0,0].set_ylabel("matched edges\n"+r"$\min{\{|E_A|,|E_B|\}}$")   
    x_pos = -.13
    all_gs_ax[0,0].annotate('', xy=(x_pos, .18), xycoords='axes fraction', xytext=(x_pos, 0.82),
                                                arrowprops=dict(arrowstyle="-", color='k'))
    all_gs_ax[1,0].set_ylabel("matched tris\n"+r"$\min{\{|T_A|,|T_B|\}}$")
    all_gs_ax[1,0].annotate('', xy=(x_pos, .225), xycoords='axes fraction', xytext=(x_pos, 0.775),
                                            arrowprops=dict(arrowstyle="-", color='k'))
    all_gs_ax[2,0].set_ylabel(r"runtime (s)")

    all_gs_ax[2,1].annotate("Klau",xy=(.075,.475),c=LT_Klau_color,xycoords='axes fraction',ha="left",fontsize=12)
    #all_gs_ax[2,1].annotate("Recursive Klau",xy=(.025,.35),c=t2_color,xycoords='axes fraction',ha="left",rotation=15)
    all_gs_ax[2,1].annotate("TAME C++\nTabu Search",xy=(.01,.77),c=T_color,xycoords='axes fraction',ha="left",fontsize=12)
    all_gs_ax[2,1].annotate("Tabu\nSearch",xy=(.65,.2),c=LT_Tabu_color,xycoords='axes fraction',ha="left",fontsize=12)


    plt.tight_layout()
    plt.show()

 