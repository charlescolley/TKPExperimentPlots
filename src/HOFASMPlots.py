from plottingStyle import * 


"""-----------------------------------------------------------------------------
                               HOFASM Experiments 
-----------------------------------------------------------------------------"""

def make_HOFASM_accuracy_plots(add_inaxis=False):
    with open(HOFASM_RESULTS+"HOFASM_seeded_accuracy_experiments_100_trials.json","r") as f:
        data = json.load(f)

    HOFASM_processed_data = []
    for n_val, experiments in sorted(data.items(),key=lambda x:float(x[0])):
        HOFASM_processed_data.append((n_val,
                               np.percentile(experiments,20),
                               np.mean(experiments),
                               np.percentile(experiments,80)))



    with open(HOFASM_RESULTS+"HOM_seeded_accuracy_experiments_100_trials.json","r") as f:
        data = json.load(f)

    HOM_processed_data = []
    for n_val, experiments in sorted(data.items(),key=lambda x:float(x[0])):
        HOM_processed_data.append((n_val,
                               np.percentile(experiments,20),
                               np.mean(experiments),
                               np.percentile(experiments,80)))

    sigmas = [.025, .05, .0725, .1, .125, .15, .175, .2]
    average_over = 100

    f = plt.figure(figsize=(350 / MY_DPI, 350 / MY_DPI), dpi=MY_DPI)
    ax = plt.gca()

    HOFASM_average_accuracies =  [acc_avg for _,_,acc_avg,_ in HOFASM_processed_data]
    HOFASM_20th_percentile_accuracies = [acc_20percentile for _, acc_20percentile, _, _ in HOFASM_processed_data]
    HOFASM_80th_percentile_accuracies = [acc_80percentile for _, _, _, acc_80percentile in HOFASM_processed_data]

    ax.fill_between(sigmas,HOFASM_20th_percentile_accuracies, HOFASM_80th_percentile_accuracies,alpha=0.3,facecolors=t4_color)
    plt.plot(sigmas, HOFASM_20th_percentile_accuracies, c=t4_color,alpha=0.4)
    plt.plot(sigmas, HOFASM_80th_percentile_accuracies, c=t4_color,alpha=0.4)
    plt.plot(sigmas, HOFASM_average_accuracies, label="HOFASM", c=t4_color)
    ax.annotate("HOFASM", xy=(.63, .81), xycoords='figure fraction',fontsize=12, c=t4_color)



    HOM_average_accuracies =  [acc_avg for _,_,acc_avg,_ in HOM_processed_data]
    HOM_20th_percentile_accuracies = [acc_20percentile for _, acc_20percentile, _, _ in HOM_processed_data]
    HOM_80th_percentile_accuracies = [acc_80percentile for _, _, _, acc_80percentile in HOM_processed_data]

    ax.fill_between(sigmas,HOM_20th_percentile_accuracies, HOM_80th_percentile_accuracies,alpha=0.1,facecolors=t1_color)
    plt.plot(sigmas, HOM_20th_percentile_accuracies, c=t1_color,alpha=0.2)
    plt.plot(sigmas, HOM_80th_percentile_accuracies, c=t1_color,alpha=0.2)
    plt.plot(sigmas, HOM_average_accuracies, label="HOM", c=t1_color)

    ax.annotate("HOM", xy=(.3, .7), xycoords='figure fraction',fontsize=12, c=t1_color)

    plt.tick_params(labelright=True)
    plt.grid(True,axis='y')
    plt.xlabel(r"stddev of Noise $\sigma$")
    plt.ylabel(f"accuracy ({average_over} trials)")

    if add_inaxis:
        axins = ax.inset_axes([.03,.06,.43,.47])
        make_HOFASM_experimentwise_accuracy_plots(ax=axins)

    plt.tight_layout()
    plt.show()


def make_HOFASM_experimentwise_accuracy_plots(ax=None):
    with open(HOFASM_RESULTS+"HOFASM_seeded_accuracy_experiments_100_trials.json","r") as f:
        HOFASM_data = json.load(f)
        HOFASM_data["0.15"].pop(27)     #corresponding experiment in HOM trial is 0.0

    with open(HOFASM_RESULTS+"HOM_seeded_accuracy_experiments_100_trials.json","r") as f:
        HOM_data = json.load(f)
        HOM_data["0.15"].pop(27)          #otherwise this throws divide by 0 error

   # return HOFASM_data, HOM_data
    processed_data = []
    for sigma, HOFASM_experiments in sorted(HOFASM_data.items(),key=lambda x:float(x[0])):
        try:
            element_wise_comparison = [x/y for x,y in zip(HOFASM_experiments,HOM_data[sigma])]
        except:
           print(sigma)
           raise
   
        processed_data.append((sigma,
                               np.percentile(element_wise_comparison,5),
                               np.percentile(element_wise_comparison,20),
  #                             min(element_wise_comparison),
                               np.percentile(element_wise_comparison,50),
  #                             np.mean(element_wise_comparison),
  #                             min(element_wise_comparison)))
                               np.percentile(element_wise_comparison, 80),
                               np.percentile(element_wise_comparison,95)))
        

    sigmas = [.025, .05, .0725, .1, .125, .15, .175, .2]
    average_over = 100

    if ax is None:
        f,ax = plt.subplots(figsize=(350 / MY_DPI, 350 / MY_DPI), dpi=MY_DPI)
        show_plot = True
    else:
        show_plot = False

    HOFASM_5th_percentile_accuracies =  list(map(lambda x:x[1], processed_data))
    HOFASM_20th_percentile_accuracies = list(map(lambda x:x[2], processed_data))
    HOFASM_median_accuracies =          list(map(lambda x: x[3], processed_data))
    HOFASM_80th_percentile_accuracies = list(map(lambda x:x[4], processed_data))
    HOFASM_95th_percentile_accuracies = list(map(lambda x:x[5], processed_data))


    ax.fill_between(sigmas,HOFASM_80th_percentile_accuracies, [1.0]*len(HOFASM_80th_percentile_accuracies),alpha=0.3,facecolors=t4_color)#,hatch="||||||")
    ax.fill_between(sigmas,HOFASM_20th_percentile_accuracies, [1.0]*len(HOFASM_20th_percentile_accuracies),alpha=0.3,facecolors=t1_color)#,hatch="\\")

    ax.plot(sigmas, HOFASM_5th_percentile_accuracies, c=t1_color,alpha=0.6)
    ax.plot(sigmas, HOFASM_95th_percentile_accuracies, c=t4_color,alpha=0.6,markersize=3)#marker='o')
    ax.plot(sigmas, HOFASM_median_accuracies, c=t4_color)
    ax.yaxis.set_ticks_position('right')
    ax.set_xticks([])


    ax.tick_params(labelright=True)
    ax.grid(True,axis='y')
    ax.set_ylim(.75,1.7)
   # plt.title("Accuracy Under Normal Perturbation")
    ax.set_xlabel(r"$\sigma$")
    ax.set_title("Accuracy Ratio",fontsize=10)#({average_over} trials)")

    #ax.set_tight_layout()
    if show_plot:
        plt.show()


def make_timing_plots():

    average_over = 100
    sigma = 1.0

    with open(HOFASM_RESULTS+"HOFASM_ourWork_runtimes_seeded_100_trials.json",'r') as f:
        data = json.load(f)

    HOFASM_new_processed_data = []
    for n_val, experiments in sorted(data.items(),key=lambda x:int(x[0])):
        HOFASM_new_processed_data.append((n_val,
                               np.percentile(experiments,20),
                               np.mean(experiments),
                               np.percentile(experiments,80)))


    with open(HOFASM_RESULTS+"HOFASM_partiallyImplicit_runtimes_seeded_100_trials.json","r") as f:
        data = json.load(f)

    HOFASM_orig_processed_data = []
    for n_val, experiments in sorted(data.items(),key=lambda x:int(x[0])):
        HOFASM_orig_processed_data.append((n_val,
                                           np.percentile(experiments,20),
                                           np.mean(experiments),
                                           np.percentile(experiments,80)))


    with open(HOFASM_RESULTS+"HOFASM_fullyImplicit_runtimes_seeded_100_trials.json","r") as f:
        data = json.load(f)

    HOFASM_orig2_processed_data = []
    for n_val, experiments in sorted(data.items(),key=lambda x:int(x[0])):
        HOFASM_orig2_processed_data.append((n_val,
                                            np.percentile(experiments,20),
                                            np.mean(experiments),
                                            np.percentile(experiments,80)))



    #convert keys to integers
    n_vals = [int(n) for n,_,_,_ in HOFASM_new_processed_data]  #should be same for all
    f = plt.figure(figsize=(350 / MY_DPI, 350 / MY_DPI), dpi=MY_DPI)
    ax = plt.gca()

    new_average_runtimes =  [t_avg for _,_,t_avg,_ in HOFASM_new_processed_data]
    new_20th_percentile_runtimes = [t_20percentile for _, t_20percentile, _, _ in HOFASM_new_processed_data]
    new_80th_percentile_runtimes = [t_80percentile for _, _, _, t_80percentile in HOFASM_new_processed_data]
    ax.fill_between(n_vals,new_20th_percentile_runtimes, new_80th_percentile_runtimes,alpha=0.2,facecolors=t4_color)

    plt.loglog(n_vals, new_20th_percentile_runtimes, label="our work", c=t4_color,alpha=0.2)
    plt.loglog(n_vals, new_80th_percentile_runtimes, label="our work", c=t4_color,alpha=0.2)
    plt.loglog(n_vals, new_average_runtimes, label="our work", c=t4_color)
    ax.annotate("our\nwork",xy=(.88, .325), xycoords='axes fraction', ha="center",fontsize=12, c=darkest_t4_color)


    #explicit marginalization results

    orig_average_runtimes =  [t_avg for _,_,t_avg,_ in HOFASM_orig_processed_data]
    orig_20th_percentile_runtimes = [t_20percentile for _, t_20percentile, _, _ in HOFASM_orig_processed_data]
    orig_80th_percentile_runtimes = [t_80percentile for _, _, _, t_80percentile in HOFASM_orig_processed_data]


    plt.loglog(n_vals, orig_20th_percentile_runtimes, label="partially implicit", c=t3_color,alpha=0.2)
    plt.loglog(n_vals, orig_80th_percentile_runtimes, label="partially implicit", c=t3_color,alpha=0.2)
    ax.fill_between(n_vals, orig_20th_percentile_runtimes, orig_80th_percentile_runtimes, alpha=0.2, facecolors=t3_color)
    plt.loglog(n_vals, orig_average_runtimes, label="partially implicit", c=t3_color)
    ax.annotate("partially\nimplicit", xy=(.375, .04), xycoords='axes fraction',ha="center",fontsize=12, c=t3_color)

    # implicit marginalization results

    orig2_average_runtimes = [t_avg for _, _, t_avg, _ in HOFASM_orig2_processed_data]
    orig2_20th_percentile_runtimes = [t_20percentile for _, t_20percentile, _, _ in HOFASM_orig2_processed_data]
    orig2_80th_percentile_runtimes = [t_80percentile for _, _, _, t_80percentile in HOFASM_orig2_processed_data]

    plt.loglog(n_vals, orig2_20th_percentile_runtimes, label="fully implicit", c=t5_color, alpha=0.2)
    plt.loglog(n_vals, orig2_80th_percentile_runtimes, label="fully implicit", c=t5_color, alpha=0.2)
    ax.fill_between(n_vals, orig2_20th_percentile_runtimes, orig2_80th_percentile_runtimes, alpha=0.2,
                    facecolors=t5_color)
    plt.loglog(n_vals, orig2_average_runtimes, label="fully implicit", c=t5_color)
    ax.annotate("fully\nimplicit", xy=(.6, .78), xycoords='axes fraction',ha="center",fontsize=12, c=t5_color)

    plt.xlabel("# source points")
    plt.ylabel("runtime (s)")

    ax.yaxis.set_ticks_position('both')
    ax.set_xticks([10,50,100])
    ax.tick_params(labeltop=False, labelright=True)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

