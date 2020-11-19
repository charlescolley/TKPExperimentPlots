from plottingStyle import * 


def RandomGraph_noise_size_experiments():

    #
    #  ER graphs p = 2log(n)/n
    #
    
    LowRankEigenAlign_ER_n_file = "LowRankEigenAlign_ErdosReyni_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    DegreeMatching_ER_n_file = "DegreeMatching_ErdosReyni_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LambdaTAME_ER_n_file = "LambdaTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    RandomMatching_ER_n_file = "RandomMatching_ErdosReyni_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_lrm_ER_n_file = "LowRankTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_lowRankMatching_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_ER_n_file = "LowRankTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"

    LowRankEigenAlign_ER_p_file = "LowRankEigenAlign_ErdosReyni_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    DegreeMatching_ER_p_file = "DegreeMatching_ErdosReyni_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LambdaTAME_ER_p_file = "LambdaTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    RandomMatching_ER_p_file ="RandomMatching_ErdosReyni_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_lrm_ER_p_file = "LowRankTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_lowRankMatching_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_ER_p_file = "LowRankTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"

    ER_results = [
        (LowRankEigenAlign_ER_p_file,LowRankEigenAlign_ER_n_file,t3_color,"LowRankEigenAlign",[(.5, .8),(.4, .6)],"ErdosReyni",":"),
        (RandomMatching_ER_p_file,RandomMatching_ER_n_file,t1_color,"RandomMatching",[(.5, .5),(.4, .3)],"ErdosReyni","-"),
        (DegreeMatching_ER_p_file,DegreeMatching_ER_n_file,t5_color,"DegreeMatching",[(.5, .6),(.4, .4)],"ErdosReyni","-"),
        (LambdaTAME_ER_p_file,LambdaTAME_ER_n_file,t2_color,"LambdaTAME",[(.5, .7),(.4, .5)],"ErdosReyni","--"),
        (LowRankTAME_lrm_ER_p_file,LowRankTAME_lrm_ER_n_file,t6_color,"LowRankTAME-lrm",[(.5, .9),(.4, .7)],"ErdosReyni","-."),
        (LowRankTAME_ER_p_file,LowRankTAME_ER_n_file,t4_color,"LowRankTAME",[(.5, .98),(.4, .8)],"ErdosReyni",(0, (3, 1, 1, 1)))
    ]

    #
    #  RG graphs degreedist = LogNormal(log(5),1)
    #
    LowRankEigenAlign_RG_n_file = "LowRankEigenAlign_RandomGeometric_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LambdaTAME_RG_n_file = "LambdaTAME_RandomGeometric_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    RandomMatching_RG_n_file = "RandomMatching_RandomGeometric_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    DegreeMatching_RG_n_file = "DegreeMatching_RandomGeometric_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_lrm_RG_n_file = "LowRankTAME_RandomGeometric_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degreedist:log5_lowRankMatching_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_RG_n_file = "LowRankTAME_RandomGeometric_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"

    DegreeMatching_RG_p_file = "DegreeMatching_RandomGeometric_degreedist:log5_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LambdaTAME_RG_p_file = "LambdaTAME_RandomGeometric_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankEigenAlign_RG_p_file = "LowRankEigenAlign_RandomGeometric_degreedist:log5_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_lrm_RG_p_file = "LowRankTAME_RandomGeometric_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_lowRankMatching_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_RG_p_file = "LowRankTAME_RandomGeometric_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    RandomMatching_RG_p_file = "RandomMatching_RandomGeometric_degreedist:log5_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"

    #RandomGeometric k = 10 degree dist results
    LowRankTAME_RG2_p_file= "RandomMatching_RandomGeometric_degreedist:k=10_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"

    RG_results = [
        (LowRankEigenAlign_RG_p_file,LowRankEigenAlign_RG_n_file,t3_color,"LowRankEigenAlign",[(.5, .8),(.4, .6)],"RandomGeometric",":"),
        (RandomMatching_RG_p_file,RandomMatching_RG_n_file,t1_color,"RandomMatching",[(.5, .5),(.4, .1)],"RandomGeometric","-"),
        (DegreeMatching_RG_p_file,DegreeMatching_RG_n_file,t5_color,"DegreeMatching",[(.5, .6),(.4, .2)],"RandomGeometric","-"),
        (LambdaTAME_RG_p_file,LambdaTAME_RG_n_file,t2_color,"LambdaTAME",[(.5, .7),(.4, .45)],"RandomGeometric","--"),
        (LowRankTAME_lrm_RG_p_file,LowRankTAME_lrm_RG_n_file,t6_color,"LowRankTAME-lrm",[(.5, .9),(.4, .7)],"RandomGeometric","-."),
        (LowRankTAME_RG_p_file,LowRankTAME_RG_n_file,t4_color,"LowRankTAME",[(.5, .98),(.4, .98)],"RandomGeometric",(0, (3, 1, 1, 1)))
    ]


    f = plt.figure(dpi=60)
    f.set_size_inches(7, 5.13)
    
    # 2 x 3 array 
    height = 0.35
    width = 0.2 
    far_left = .125
    lower_bottom = .125
    upper_bottom = .55
    size_plot_offset = .15

    rectangle1 = [far_left, lower_bottom, width, height]
    rectangle2 = [far_left, upper_bottom, width, height]


    ER_noise_ax = [plt.axes(rectangle1),plt.axes(rectangle2)]
    for ax in ER_noise_ax:
        ax.set_yticks([0.0,.25,.5,.75,1.0])


    ER_noise_ax[0].set_xlabel(r"$q$")
    ER_noise_ax[0].set_ylabel("accuracy")
    ER_noise_ax[1].set_ylabel(r"matched tris$/ \min{\{|T_A|,|T_B|\}}$")
    ER_noise_ax[1].set_title("ER")

    for (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in ER_results:
        
        TAME_random_graph_noise_experiments(ER_noise_ax, p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle)
    

    rectangle1 = [far_left+width+.02, lower_bottom, width, height]
    rectangle2 = [far_left+width+.02, upper_bottom, width, height]

    
    RG_noise_ax = [plt.axes(rectangle1, sharey=ER_noise_ax[0]),plt.axes(rectangle2, sharey=ER_noise_ax[0])]
    for ax in RG_noise_ax:
        ax.yaxis.set_ticks_position('right')
        xgridlines = ax.get_xgridlines()
        xgridlines[-1].set_color('r')
        xgridlines[-1].set_linestyle((0, (5, 10)))
        ax.yaxis.set_ticks_position('right')

    RG_noise_ax[0].set_xlabel(r"$q$")
    RG_noise_ax[1].set_title("RG")
    RG_noise_ax[1].annotate("LR-TAME",xy=(.55, .9), xycoords='axes fraction', c=darker_t4_color)

    for (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in RG_results:
        TAME_random_graph_noise_experiments(RG_noise_ax, p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle)


    rectangle1 = [far_left+2*width+size_plot_offset, lower_bottom, width, height]
    rectangle2 = [far_left+2*width+size_plot_offset, upper_bottom, width, height]
    
    RG_sizes_ax = [plt.axes(rectangle1, sharey=ER_noise_ax[0]),plt.axes(rectangle2,sharey=ER_noise_ax[0])]
    for ax in RG_sizes_ax:
        ax.yaxis.set_ticks_position('right')

    RG_sizes_ax[1].set_title("RG")
    for (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in RG_results:
        TAME_random_graph_size_experiments(RG_sizes_ax, p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle)
    
    #put labels on the plots 
    RG_sizes_labels = [ 
        (t6_color,"LR-TAME-lrm",(.25, .75)),
        (darker_t4_color,"LR-TAME",(.1, .875)),
        (t2_color,r"$\Lambda-$TAME",(.4, .37))

    ]
    
    for (c,text, loc) in RG_sizes_labels:
        RG_sizes_ax[1].annotate(text,xy=loc, xycoords='axes fraction', c=c)

    RG_noise_labels = [
       (t1_color,"Random",(.08, .05)),
       (t5_color,"Degree",(.08, .3))
    ]
    for (c,text,loc) in [(t2_color,r"$\Lambda-$TAME",(.4, .575)),(t3_color,"LR-EigenAlign",(.2, .15))]:
        RG_sizes_ax[0].annotate(text,xy=loc, xycoords='axes fraction', c=c)

    for (c,text, loc) in RG_noise_labels:
        RG_noise_ax[0].annotate(text,xy=loc, xycoords='axes fraction', c=c)
    
  
    plt.show()


def TAME_random_graph_size_experiments(axes, p_file,n_file,color,anotate_label,anotate_loc,graph_type,linestyle="solid"): 
    #
    #  Load the files
    #

    if graph_type == "RandomGeometric":
        def load_data(files):
            data = []
            for file in files:
                with open(TAME_RESULTS +"RandomGraphAccuracyExperiments/" + file,"r") as f:
                    data.extend(json.load(f))
            return data
    elif graph_type == "ErdosReyni":
        def load_data(files):
            data = []
            for file in files:
                with open(TAME_RESULTS + "RandomGraphAccuracyExperiments/" + file,"r") as f:
                    data.extend(json.load(f))
            return data
    else: 
        raise ValueError(f"graph type:{graph_type} must be one of 'RandomGeometric' or 'ErdosReyni'")

    
    n_data = load_data([n_file])

    #
    #  Parse the data
    #

    def process_data_for_sizes(data):
        accuracies = {}
        dw_accuracies = {}
        triangle_match = {}
        tri_to_acc_data = []

        for (seed,p_remove,n,acc,degree_weighted_acc,matched_tri,A_tri,B_tri,_)in data:

            if n in accuracies:
                accuracies[n].append(acc)
            else:  
                accuracies[n] = [acc] 

            if n in dw_accuracies:
                dw_accuracies[n].append(degree_weighted_acc)
            else:  
                dw_accuracies[n] = [degree_weighted_acc] 

            if n in triangle_match:
                 triangle_match[n].append(matched_tri/min(A_tri,B_tri))
            else:  
                triangle_match[n] = [matched_tri/min(A_tri,B_tri)]

            
            tri_to_acc_data.append((abs(A_tri - B_tri),acc))

        return accuracies, dw_accuracies, triangle_match, tri_to_acc_data
    
    n_acc, n_dw_acc, n_tri_match, tri_to_acc_data2 = process_data_for_sizes(n_data)
    

    
    #
    #  Plot Size experiments
    #
    # -------------------------  Accuracies   -------------------------- #
    
    ax = axes[0]
    ax.set_xlim(100,2000)
    ax.set_yticks([0.0,.25,.5,.75,1.0])
    ax.set_xticks([100,1000,2000])
    #ax.set_xlabel(r"n")
    #ax.set_ylabel("accuracy")
    
    ax.grid(which="major", axis="both",alpha=.3)
    ax.set_xlabel(r"n")
    ax.spines['left'].set_color('r')
    ax.spines['left'].set_linestyle((0, (5, 10)))

    def make_percentile_plot(plot_ax, data,color,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
 #           (30,70,.2,color)
            (20,80,.2,color)
        ]

        sizes = list(data.keys())
        perm = sorted(range(len(sizes)),key=lambda i:sizes[i])
        sizes = np.array([sizes[i] for i in perm])

        x = list(data.values())
        size_exp_data = np.array([x[i] for i in perm]).T

        plot_percentiles(plot_ax,  size_exp_data, sizes, lines, ribbons,**kwargs)



    make_percentile_plot(ax,n_acc,color,linestyle=linestyle)
    
    # ---------------  Tri - Match Accuracies   ---------------- #

    ax = axes[1]
    ax.set_xlim(100,2000)
    ax.set_yticks([0.0,.25,.5,.75,1.0])
    ax.set_xticks([100,1000,2000])
    #ax.set_ylabel(r"$\frac{|T_{M(A,B)}|}{\min{|T_A|,|T_B|}}$")
    
    ax.grid(which="major", axis="both",alpha=.3)
    #ax.yaxis.set_ticks_position('right')
    ax.spines['left'].set_color('r')
    ax.spines['left'].set_linestyle((0, (5, 10)))
 #   ax.set_yscale("lo")

    make_percentile_plot(ax,n_tri_match,color,linestyle=linestyle)

def TAME_random_graph_noise_experiments(axes,p_file,n_file,color,anotate_label,anotate_loc,graph_type,linestyle="solid"):

    #
    #  Load the files
    #

    if graph_type == "RandomGeometric":
        def load_data(files):
            data = []
            for file in files:
                with open(TAME_RESULTS + "RandomGraphAccuracyExperiments/" + file,"r") as f:
                    data.extend(json.load(f))
            return data
    elif graph_type == "ErdosReyni":
        def load_data(files):
            data = []
            for file in files:
                print(file)
                with open(TAME_RESULTS + "RandomGraphAccuracyExperiments/"+ file,"r") as f:
                    data.extend(json.load(f))
            return data
    else: 
        raise ValueError(f"graph type:{graph_type} must be one of 'RandomGeometric' or 'ErdosReyni'")

    p_data = load_data([p_file])
    n_data = load_data([n_file])


    #
    #  Parse the data
    #

    
    def process_data_for_noise(data):
        accuracies = {}
        dw_accuracies = {}
        triangle_match = {}
        tri_to_acc_data = []

        for (seed,p_remove,n,acc,degree_weighted_acc,matched_tri,A_tri,B_tri,_)in data:

            if p_remove in accuracies:
                accuracies[p_remove].append(acc)
            else:  
                accuracies[p_remove] = [acc] 
            
            if p_remove in dw_accuracies:
                dw_accuracies[p_remove].append(degree_weighted_acc)
            else:  
                dw_accuracies[p_remove] = [degree_weighted_acc] 

            if p_remove in triangle_match:
                 triangle_match[p_remove].append(matched_tri/min(A_tri,B_tri))
            else:  
                triangle_match[p_remove] = [matched_tri/min(A_tri,B_tri)]

            tri_to_acc_data.append((abs(A_tri - B_tri),acc))

        return accuracies, dw_accuracies, triangle_match, tri_to_acc_data

    def process_data_for_sizes(data):
        accuracies = {}
        dw_accuracies = {}
        triangle_match = {}
        tri_to_acc_data = []

        for (seed,p_remove,n,acc,degree_weighted_acc,matched_tri,A_tri,B_tri,_)in data:

            if n in accuracies:
                accuracies[n].append(acc)
            else:  
                accuracies[n] = [acc] 

            if n in dw_accuracies:
                dw_accuracies[n].append(degree_weighted_acc)
            else:  
                dw_accuracies[n] = [degree_weighted_acc] 

            if n in triangle_match:
                 triangle_match[n].append(matched_tri/min(A_tri,B_tri))
            else:  
                triangle_match[n] = [matched_tri/min(A_tri,B_tri)]

            
            tri_to_acc_data.append((abs(A_tri - B_tri),acc))

        return accuracies, dw_accuracies, triangle_match, tri_to_acc_data    


    p_acc, p_dw_acc, p_tri_match, tri_to_acc_data = process_data_for_noise(p_data)





    #
    #  Plot Noise experiments
    #
    # -------------------------  Accuracies   -------------------------- #
    ax = axes[0]

    ax.set_ylim(0,1.0)
    ax.set_xlim(0,.2)
    ax.set_xscale("log")
    ax.grid(which="major", axis="both",alpha=.3)
    
    def make_percentile_plot(plot_ax, data,color,linestyle):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
  #          (30,70,.2,color)
            (20,80,.2,color)
        ]

        noise_levels = list(data.keys())
        perm = sorted(range(len(noise_levels)),key=lambda i:noise_levels[i])
        noise_levels = np.array([noise_levels[i] for i in perm])

        data = list(data.values())
        data = np.array([data[i] for i in perm]).T

        plot_percentiles(plot_ax,  data, noise_levels, lines,ribbons,linestyle=linestyle)

    make_percentile_plot(ax,p_acc,color,linestyle=linestyle)

    # ------------------------  Tri Match Ratios   ---------------------#
    
    ax = axes[1]
    ax.set_ylim(0,1.0)
    ax.set_xlim(0,.2)

    ax.set_xscale("log")
    ax.grid(which="major", axis="both",alpha=.3)

    make_percentile_plot(ax,p_tri_match,color,linestyle=linestyle)


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

    print(len(TAME_nonzero_second_largest_sing_vals))
    print(len(TAME_zero_second_largest_sing_vals))
    
    LowRankTAME_nonzero_second_largest_sing_vals, LowRankTAME_nonzero_vertex_products, \
        LowRankTAME_nonzero_triangle_products, LowRankTAME_zero_second_largest_sing_vals,\
             LowRankTAME_zero_vertex_products, LowRankTAME_zero_triangle_products = process_data(LowRankTAME_data)

    """
    #print(LowRankTAME_nonzero_second_largest_sing_vals)
    #print(LowRankTAME_zero_second_largest_sing_vals)
    ax = plt.subplot(121)

    #format the axis
    ax.set_yscale("log")
    ax.grid(which="major", axis="y")
    ax.annotate("TAME", xy=(.1, .9), xycoords='axes fraction', c=t1_color,size=10)
    ax.annotate("LowRankTAME", xy=(.1, .1), xycoords='axes fraction', c=darker_t4_color,size=10)
    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=t3_color)
#    ax.set_xscale("log")
    ax.set_ylabel(r"$\sigma_2$")
    #ax.set_ylabel(r"max $\sigma_2")
    ax.set_xlabel(r"|$V_A$||$V_B$|")
    ax.set_ylim(1e-16,1e-7)
    ax.set_xlim(1e2,1.5e8)
    plt.xticks([1e7,1e8])
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())

    #scatter plot formatting
    marker_size = 20
    marker_alpha = 1.0

    #plot TAME results
    plt.scatter(TAME_nonzero_vertex_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=t1_color,s=marker_size,alpha=marker_alpha)
    plt.scatter(TAME_zero_vertex_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=t1_color,s=marker_size,alpha=marker_alpha)
    
    #plot machine epsilon
    all_vertex_products = list(itertools.chain(TAME_zero_vertex_products,TAME_nonzero_vertex_products))
    #plt.plot(sorted(all_vertex_products),[2e-16]*(len(all_vertex_products)),c=t3_color)
    plt.plot([0,1.5e8],[2e-16]*2,c=t3_color,zorder=1)

    #plot LowRankTAME results
    plt.scatter(LowRankTAME_nonzero_vertex_products,LowRankTAME_nonzero_second_largest_sing_vals,c=darker_t4_color,marker='o',s=marker_size,alpha=marker_alpha,zorder=2)
    plt.scatter(LowRankTAME_zero_vertex_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=darker_t4_color,s=marker_size,alpha=marker_alpha,zorder=2)
    #print(LowRankTAME_zero_vertex_products)
    #print(LowRankTAME_zero_second_largest_sing_vals)
    
    #make subwindow
    #axins = inset_axes(ax, width=1.3, height=0.9)
    """

    ax = axes
#    ax = plt.subplot(122)
    ax.set_yscale("log")
    ax.grid(which="major", axis="y")
    #ax.set_ylabel(r"max $\sigma_2")
    #ax.set_ylabel(r"max [$\sum_{i=2}^k\sigma_i]")
    #ax.yaxis.set_ticks_position('right')
    #ax.tick_params(labeltop=False, labelright=True)
    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=t3_color)
    ax.annotate("TAME", xy=(.2,.7),xycoords='axes fraction',fontsize=12,c=t1_color)
    ax.annotate("LowRankTAME", xy=(.2,.1),xycoords='axes fraction',fontsize=12,c=t4_color)
    ax.set_ylabel(r"$\sigma_2$")
    ax.set_xlabel(r"|$T_A$||$T_B$|")
    ax.set_yscale("log")
    ax.set_xscale("log")

    
    ax.set_xlim(1e5,1e13)
    ax.set_ylim(1e-16,1e-7)

   #scatter plot formatting
    marker_size = 15
    marker_alpha = 1.0

    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o',c=t1_color,s=marker_size)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=t1_color,s=marker_size)

    #plot machine epsilon
    #all_triangle_products = list(itertools.chain(TAME_zero_triangle_products,TAME_nonzero_triangle_products)) 
    #plt.plot(sorted(all_triangle_products),[2e-16]*(len(all_triangle_products)),c=t3_color)
    plt.plot([1e5,1e13],[2e-16]*2,c=t3_color,zorder=1)

    plt.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=darker_t4_color,s=marker_size)
    scatter = plt.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=darker_t4_color,s=marker_size,zorder=2)
    #print(type(scatter))
   # legend_elements =  [plt.scatter([], [], marker='o', color='k', label='zero', markerfacecolor='none',markersize=5),
    #                    plt.scatter([],[], marker="*", ms=10)]
   # ax.legend(handles=legend_elements, loc='upper left')
    

    #TODO:combining plots, potential remove later
    plt.subplots_adjust(bottom=0.2,top=.95,left=.17)
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
            """
            sing_vals = [(i,s[1]) if len(s) > 1 else (i,2e-16) for i,s in enumerate(profile_dict["sing_vals"])]
            print(profile_dict["ranks"])
            for (i,s) in sing_vals:
                if profile_dict["ranks"][i] > 1: 
                    #print([sum(s) for s in profile_dict["sing_vals"]])
                    nonzero_second_largest_sing_vals.append(min(sing_vals))
                    nonzero_vertex_products.append(n**2)
                    nonzero_triangle_products.append(max_tris**2) 

            """

        return n_values, nonzero_second_largest_sing_vals, nonzero_vertex_products, nonzero_triangle_products, zero_second_largest_sing_vals, zero_vertex_products, zero_triangle_products


    n_values, LowRankTAME_nonzero_second_largest_sing_vals, LowRankTAME_nonzero_vertex_products,\
         LowRankTAME_nonzero_triangle_products, LowRankTAME_zero_second_largest_sing_vals,\
              LowRankTAME_zero_vertex_products, LowRankTAME_zero_triangle_products =\
                   process_RandomGeometricResults(LowRankTAME_data)
    _, TAME_nonzero_second_largest_sing_vals, TAME_nonzero_vertex_products,\
         TAME_nonzero_triangle_products, TAME_zero_second_largest_sing_vals,\
              TAME_zero_vertex_products, TAME_zero_triangle_products =\
                   process_RandomGeometricResults(TAME_data)

    print(len(TAME_nonzero_second_largest_sing_vals))
    print(len(TAME_zero_second_largest_sing_vals))
    
    

    #return LowRankTAME_data 
    """
    #
    #   Make Vertex_Vertex plots
    #
    ax = plt.subplot(121)

    #format the axis
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(which="major", axis="y")
    ax.grid(which="minor", axis="x")
    #ax.annotate("TAME", xy=(.1, .9), xycoords='axes fraction', c=darkest_t4_color,size=10)
    #ax.annotate("LowRankTAME", xy=(.1, .1), xycoords='axes fraction', c=t2_color,size=10)
    ax.annotate(r"$\epsilon$", xy=(1.06, .02), xycoords='axes fraction', c=t3_color)
#    ax.set_xscale("log")
    ax.set_ylabel(r"$\sigma_2$")
    #ax.set_ylabel(r"max $\sigma_2")
    ax.set_xlabel(r"|$V_A$||$V_B$|")
 #   ax.set_ylim(1e-16,1e-7)
    ax.set_xlim(5e3,7e8)
    ax.set_xticks([1e4,1e6,1e7,1e8])
    print(n_values)
    print([n**2 for n in n_values])
    #ax.set_xticks([2e5,4e8])
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())

    #scatter plot formatting
    marker_size = 20
    marker_alpha = 1.0



    #plot machine epsilon
    #all_vertex_products = list(itertools.chain(TAME_zero_vertex_products,TAME_nonzero_vertex_products))
    #plt.plot(sorted(all_vertex_products),[2e-16]*(len(all_vertex_products)),c=t3_color)
    plt.plot([1e3,1e9],[2e-16]*2,c=t3_color,zorder=1)

    #plot LowRankTAME results
    plt.scatter(LowRankTAME_nonzero_vertex_products,LowRankTAME_nonzero_second_largest_sing_vals,c=darker_t4_color,marker='o',s=marker_size,alpha=marker_alpha,zorder=2)
    plt.scatter(LowRankTAME_zero_vertex_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=darker_t4_color,s=marker_size,alpha=marker_alpha,zorder=2)
    #print(LowRankTAME_zero_vertex_products)
    #print(LowRankTAME_zero_second_largest_sing_vals)
    
    #plot TAME results 
    plt.scatter(TAME_nonzero_vertex_products,TAME_nonzero_second_largest_sing_vals,c=t1_color,marker='o',s=marker_size,alpha=marker_alpha,zorder=2)
    plt.scatter(TAME_zero_vertex_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=t1_color,s=marker_size,alpha=marker_alpha,zorder=2)
    
    #make zoom in window
    axins = ax.inset_axes([.63,.1,.15,.45]) # zoom = 6
    axins.scatter(TAME_nonzero_vertex_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=t1_color,s=marker_size)
    axins.scatter(TAME_zero_vertex_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=t1_color,s=marker_size)
    # sub region of the original image
#    axins.set_xlim(9e9, 3e11)
    axins.set_xlim(6e7, 2e8)
    axins.set_ylim(7e-13, 6e-12)
    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_xticks([])
    axins.minorticks_off()
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5",alpha=.5,zorder=1)
    """


    #
    #   Make Triangle_Triangle plots
    #
    ax = axes
#    ax = plt.subplot(122)

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
    plt.scatter(LowRankTAME_nonzero_triangle_products,LowRankTAME_nonzero_second_largest_sing_vals,marker='o', c=darker_t4_color,s=marker_size,zorder=2)
    plt.scatter(LowRankTAME_zero_triangle_products,LowRankTAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=darker_t4_color,s=marker_size,zorder=2)

    #plot TAME Data
    plt.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=t1_color,s=marker_size,zorder=3)
    plt.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=t1_color,s=marker_size,zorder=3)


    axins = ax.inset_axes([.6,.15,.25,.25]) # zoom = 6
    axins.scatter(TAME_nonzero_triangle_products,TAME_nonzero_second_largest_sing_vals,marker='o', c=t1_color,s=marker_size)
    axins.scatter(TAME_zero_triangle_products,TAME_zero_second_largest_sing_vals,facecolors='none',edgecolors=t1_color,s=marker_size)
    # sub region of the original image
#    axins.set_xlim(9e9, 3e11)
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



def make_LVGNA_runtime_performance_plots():

    f = plt.figure(dpi=60)
    f.set_size_inches(8, 5)

    height = 0.8
    width = 0.55 
    far_left = .08
    bottom = .125
    pad = .08

    rectangle1 = [far_left, bottom, width, height]
    rectangle2 = [far_left+width+pad, bottom, .25, height]

    axes = [plt.axes(rectangle1),plt.axes(rectangle2)]

    make_LVGNA_runtime_plots(axes[0])
    make_LVGNA_performance_plots(axes[1])

    plt.show()


#Original Performance plots, currently supporting LambdaTAME, TAME (C++), LGRAAL, and LowRankTAME 
#TODO: must be updated to have, TAME (Julia (?)), max rank LowRankTAME implementations
def make_LVGNA_performance_plots(ax=None):

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

    n = len(graph_names)
    LGRAAL_performance = []
    LGRAAL_accuracy = []
    TAME_performance = []
    TAME_accuracy = []
    LambdaTAME_performance = []
    LambdaTAME_accuracy = []
    LowRankTAME_performance = []
    LowRankTAME_accuracy = []
    LowRankTAME_LRMatch_performance = []
    LowRankTAME_LRMatch_accuracy = []
    LowRankEigenAlign_performance = []
    LowRankEigenAlign_accuracy = []

    Is,Js = np.triu_indices(n,k=1)
    for i,j in zip(Is,Js):

        best = max(LambdaTAME_results[i,j],Original_TAME_tri_results[i,j],LGRAAL_tri_results[i,j],LowRankTAME_results[i,j],LowRankTAME_LRMatch_results[i,j],LowRankEigenAlign_results[i,j])

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

    

   
    #for i = 1:size(c, 2)
    #plot!(t, [sum(R[:,i] .<= ti)/size(c,1) for ti in t], label="Alg $i", t=:steppre, lw=2)
    #end

    
    #ax = [ax] #jerry rigged
    ax.plot(range(len(TAME_performance)), TAME_performance, label="TAME", c=t1_color)
    #bold_outlined_text(ax,"TAME (C++)",t1_color,(.82, .9))
    ax.annotate("TAME (C++)",xy=(.6, .77), xycoords='axes fraction', c=t1_color,rotation=-35)
   

    ax.plot(range(len(LambdaTAME_performance)), LambdaTAME_performance, label="$\Lambda$-TAME", c=t2_color,linestyle="--")
    ax.annotate("$\Lambda$-TAME",xy=(.5, .78), xycoords='axes fraction', c=t2_color,rotation=-35)
    
    ax.plot(range(len(LGRAAL_performance)), LGRAAL_performance, label="LGRAAL", c=t5_color,linestyle=(0,(3,1,1,1,1,1)))
    ax.annotate("L-GRAAL",xy=(.47, .46), xycoords='axes fraction', c=t5_color)
    
    ax.plot(range(len(LowRankTAME_performance)), LowRankTAME_performance, label="LowRankTAME", c=t4_color,linestyle=(0, (3, 1, 1, 1)))
    ax.annotate("LowRankTAME",xy=(.4, .96), xycoords='axes fraction', c=t4_color)
    
    ax.plot(range(len(LowRankTAME_LRMatch_performance)), LowRankTAME_LRMatch_performance, label="LowRankTAME", c=t6_color,linestyle="-.")
    ax.annotate("LowRankTAME-(lrm)",xy=(.35, .57),xycoords='axes fraction', c=t6_color,rotation=-45)
    
    ax.plot(range(len(LowRankEigenAlign_performance)),LowRankEigenAlign_performance,label="LowRankEigenAlign", c=t3_color,linestyle=":")
    ax.annotate("LowRankEigenAlign",xy=(.1, .1),xycoords='axes fraction', c=t3_color,rotation=-48)
    
    
    ax.set_ylabel("performance ratio")
    ax.grid(which="both")
    ax.set_xlabel("experiment order")
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

def make_LVGNA_runtime_plots(ax=None):

    graph_names, LGRAAL_tri_results, LGRAAL_runtimes, \
        Original_TAME_tri_results, TAME_runtimes, LambdaTAME_runtimes,\
             new_TAME_tri_results = get_results()
    

    def process_LowRankTAME_data(f):
        _, results = json.load(f)

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



    n = len(graph_names)
    problem_sizes = []

    LGRAAL_exp_runtimes = []
    TAME_exp_runtimes = []
    LambdaTAME_exp_runtimes = []
    LowRankTAME_exp_runtimes = []
    LowRankTAME_LRM_exp_runtimes = []
    LowRankEigenAlign_exp_runtimes = []

    Is,Js = np.triu_indices(n,k=1)
    for i,j in zip(Is,Js):

        LGRAAL_exp_runtimes.append(LGRAAL_runtimes[i,j])
        TAME_exp_runtimes.append(TAME_runtimes[i,j])
        LambdaTAME_exp_runtimes.append(LambdaTAME_runtimes[i,j])
        LowRankTAME_exp_runtimes.append(LowRankTAME_runtimes[i, j])
        LowRankTAME_LRM_exp_runtimes.append(LowRankTAME_LRM_runtimes[i, j])
        LowRankEigenAlign_exp_runtimes.append(LowRankEigenAlign_runtimes[i,j])

        problem_sizes.append(triangle_counts[graph_names[i]]*triangle_counts[graph_names[j]])


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
    ax.scatter(problem_sizes,TAME_exp_runtimes,label="TAME", c=t1_color,marker='s')
    plot_1d_loess_smoothing(problem_sizes,TAME_exp_runtimes,loess_smoothing_frac,ax,c=t1_color,linestyle="solid")
#    ax[0].plot(range(len(old_TAME_performance)), old_TAME_performance, label="TAME", c=t4_color)
    ax.annotate("TAME (C++)",xy=(.53, .85), xycoords='axes fraction', c=t1_color)


#    print(new_TAME_exp_runtimes)
    ax.scatter(problem_sizes,LambdaTAME_exp_runtimes,label="$\Lambda$-TAME", c=t2_color,marker='^')
    plot_1d_loess_smoothing(problem_sizes,LambdaTAME_exp_runtimes,loess_smoothing_frac,ax,c=t2_color,linestyle="--")
    #ax[0].plot(range(len(new_TAME_performance)), new_TAME_performance, label="$\Lambda$-TAME", c=t2_color)
    ax.annotate("$\Lambda$-TAME",xy=(.65, .17), xycoords='axes fraction', c=t2_color)
    
    ax.scatter(problem_sizes, LGRAAL_exp_runtimes,label="$LGRAAL", c=t5_color,zorder=-1,marker='x')
    plot_1d_loess_smoothing(problem_sizes,LGRAAL_exp_runtimes,loess_smoothing_frac,ax,c=t5_color,linestyle=(0,(3,1,1,1,1,1)))
    #ax.plot(range(len(LGRAAL_performance)), LGRAAL_performance, label="LGRAAL", c=t1_color)
    ax.annotate("L-GRAAL",xy=(.2, .7), xycoords='axes fraction', c=t5_color)

    ax.scatter(problem_sizes,LowRankTAME_exp_runtimes,label="LowRankTAME", c=t4_color)
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_exp_runtimes,loess_smoothing_frac,ax,c=t4_color,linestyle=(0, (3, 1, 1, 1)))
    ax.annotate("LowRankTAME",xy=(.7, .57), xycoords='axes fraction', c=t4_color)
 
    ax.scatter(problem_sizes,LowRankTAME_LRM_exp_runtimes,facecolors='none',edgecolors=t6_color,label="LowRankTAME-(lrm)")
    plot_1d_loess_smoothing(problem_sizes,LowRankTAME_LRM_exp_runtimes,loess_smoothing_frac,ax,c=t6_color,linestyle="-.")
    ax.annotate("LowRankTAME-(lrm)",xy=(.55, .48), xycoords='axes fraction', c=t6_color)
 
    ax.scatter(problem_sizes,LowRankEigenAlign_exp_runtimes,label="LowRankEigenAlign",c=t3_color,marker="*")
    plot_1d_loess_smoothing(problem_sizes,LowRankEigenAlign_exp_runtimes,loess_smoothing_frac,ax,c=t3_color,linestyle=":")
    ax.annotate("LowRankEigenAlign",xy=(.1, .27), xycoords='axes fraction', c=t3_color)
    
 

    """
    ax[1].plot(range(len(old_TAME_accuracy)),old_TAME_accuracy,label="TAME", c=t5_color)
    ax[1].plot(range(len(new_TAME_accuracy)),new_TAME_accuracy, label="$\Lambda$-TAME", c=t2_color)
    ax[1].plot(range(len(LGRAAL_accuracy)),LGRAAL_accuracy,label="LGRAAL", c=t1_color)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("experiment rank")
    ax[1].grid(which="both")
    """
    #plt.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()

def max_rank_experiments():


    f = plt.figure(dpi=60)
    f.set_size_inches(5, 3.5)
    f, axes = plt.subplots(2,1)
    #-------------------------------Plot Synth Data--------------------------------------

    with open(TAME_RESULTS + "MaxRankExperiments/LowRankTAME_RandomGeometric_degreedist_log5_iter:15_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100,500,1K,2K,5K,10K,20K]_noMatching_pRemove:[.01,.05]_tol:1e-12_trialcount:50.json","r") as f:
        synth_results = json.load(f)
    
    MM_exp_data, n_vals, p_vals, param_vals, tri_counts = process_synthetic_TAME_output2(synth_results)
 
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_ylim(00,315)

    label_meta = [
        ((.85, .07),"o",t1_color),
        ((.85, .22),"v",t2_color),
        ((.7, .475),"*",t3_color),
        ((.83, .85),"s",t4_color)]#  no_ylim_points
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
        #ax.plot(marker_loc[0],marker_loc[1],marker,c=c,markersize=5)


    #ax.set_xticks(list(n_vals.keys()))
    ax.set_title("Synthetic graphs")
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=True)
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    #ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.grid(which="major", axis="y")
    #ax.set_xlim(3e6,2e11)

    ax.set_xticklabels([1e6,1e7,1e8,1e9,1e10,1e11,1e12])
    ax.set_xlim(1e5,1e12)
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    ax.set_ylabel("maximum rank")
    ax.set_yticks([50,100,150,200,250,300])
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
    label_meta = [
        ((.05, .35),"o",t1_color),
        ((.05, .59),"v",t2_color),
        ((.1, .8),"*",t3_color),
        ((.09, .1),"s",t4_color)]

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
#        plt.plot(xout,yout,c=c,linestyle="--")

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
    plt.tight_layout()
    plt.show()
