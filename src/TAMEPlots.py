from plottingStyle import * 


def LowRankEigenAlign_noise_size_experiments():
    #Old Files
    """
    LowRankEigenAlign_RG_p_file = "LowRankEigenAlign_RandomGeometric_degreedist:log5_n:[50]_p_remove:[.00:.01:.05]_trialcount:50.json"    
    LowRankEigenAlign_ER_p_file = "LowRankEigenAlign_ErdosReyni_n:[50]_p_remove:[.00:.01:.05]_trialcount:50.json"
    LowRankEigenAlign_ER_n_file = "LowRankEigenAlign_ErdosReyni_n:[50,100,250,500,1K,2K]_p_remove:[.01,.05,.1]_trialcount:20.json"
    LowRankEigenAlign_RG_n_file = "LowRankEigenAlign_RandomGeometric_degreedist:log5_n:[50,100,250,500,1K,2K]_p_remove:[.01,.05,.1]_trialcount:20.json"

    results = [
        (LowRankEigenAlign_RG_p_file,LowRankEigenAlign_RG_n_file,t2_color,"Random Geometric",[(.3, .8),(.3, .8)],"RandomGeometric"),
        (LowRankEigenAlign_ER_p_file,LowRankEigenAlign_ER_n_file,t3_color,"Erdos Reyni",[(.2, .1),(.2,.1)],"ErdosReyni")
    ]


    """
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
    #  ER graphs p = 10log(n)/n
    #
    """
    DegreeMatching_ER2_n_file = "DegreeMatching_ErdosReyni_pedges:10logn_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_lrm_ER2_n_file = "LowRankTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_lowRankMatching_pedges:10logn_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankEigenAlign_ER2_n_file = "LowRankEigenAlign_ErdosReyni_pedges:10logn_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LambdaTAME_ER2_n_file = "LambdaTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_pedges:10logn_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    RandomMatching_ER2_n_file = "RandomMatching_ErdosReyni_pedges:10logn_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_ER2_n_file = "LowRankTAME_ErdosReyni_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_pedges:10logn_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"

    DegreeMatching_ER2_p_file = "DegreeMatching_ErdosReyni_pedges:10logn_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LambdaTAME_ER2_p_file = "LambdaTAME_ErdosReyni_pedges:10logn_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankEigenAlign_ER2_p_file ="LowRankEigenAlign_ErdosReyni_pedges:10logn_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_lrm_ER2_p_file = "LowRankTAME_ErdosReyni_pedges:10logn_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_lowRankMatching_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_ER2_p_file = "LowRankTAME_ErdosReyni_pedges:10logn_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    RandomMatching_ER2_p_file = "RandomMatching_ErdosReyni_pedges:10logn_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"

    ER2_results = [
        (LowRankEigenAlign_ER2_p_file,LowRankEigenAlign_ER2_n_file,t3_color,"LowRankEigenAlign",[(.5, .8),(.4, .6)],"ErdosReyni",":"),
        (RandomMatching_ER2_p_file,RandomMatching_ER2_n_file,t1_color,"RandomMatching",[(.5, .5),(.4, .3)],"ErdosReyni","-"),
        (DegreeMatching_ER2_p_file,DegreeMatching_ER2_n_file,t5_color,"DegreeMatching",[(.5, .6),(.4, .4)],"ErdosReyni","-"),
        (LambdaTAME_ER2_p_file,LambdaTAME_ER2_n_file,t2_color,"LambdaTAME",[(.5, .7),(.4, .5)],"ErdosReyni","--"),
        (LowRankTAME_lrm_ER2_p_file,LowRankTAME_lrm_ER2_n_file,t6_color,"LowRankTAME-lrm",[(.5, .9),(.4, .7)],"ErdosReyni","-."),
        (LowRankTAME_ER2_p_file,LowRankTAME_ER2_n_file,t4_color,"LowRankTAME",[(.5, .98),(.4, .8)],"ErdosReyni",(0, (3, 1, 1, 1)))
    ]
    """

    #
    #  ER graphs degreedist = LogNormal(log(5),1)
    #
    """
    DegreeMatching_ER3_n_file = "DegreeMatching_ErdosReyni_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_lrm_ER3_n_file = "LowRankTAME_ErdosReyni_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_lowRankMatching_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankEigenAlign_ER3_n_file = "LowRankEigenAlign_ErdosReyni_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LambdaTAME_ER3_n_file = "LambdaTAME_ErdosReyni_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    RandomMatching_ER3_n_file = "RandomMatching_ErdosReyni_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"
    LowRankTAME_ER3_n_file = "LowRankTAME_ErdosReyni_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove:[.01]_trialcount:20.json"


    DegreeMatching_ER3_p_file = "DegreeMatching_ErdosReyni_degreedist:log5_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LambdaTAME_ER3_p_file = "LambdaTAME_ErdosReyni_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankEigenAlign_ER3_p_file ="LowRankEigenAlign_ErdosReyni_degreedist:log5_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_lrm_ER3_p_file = "LowRankTAME_ErdosReyni_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_lowRankMatching_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    LowRankTAME_ER3_p_file = "LowRankTAME_ErdosReyni_degreedist:log5_alphas:[.5,1.0]_betas:[0.0,1e0,1e1,1e2]_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    RandomMatching_ER3_p_file = "RandomMatching_ErdosReyni_degreedist:log5_n:[100]_p_remove:[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"

    
    ER3_results = [
        (LowRankEigenAlign_ER3_p_file,LowRankEigenAlign_ER3_n_file,t3_color,"LowRankEigenAlign",[(.5, .8),(.4, .6)],"ErdosReyni",":"),
        (RandomMatching_ER3_p_file,RandomMatching_ER3_n_file,t1_color,"RandomMatching",[(.5, .5),(.4, .3)],"ErdosReyni","-"),
        (DegreeMatching_ER3_p_file,DegreeMatching_ER3_n_file,t5_color,"DegreeMatching",[(.5, .6),(.4, .4)],"ErdosReyni","-"),
        (LambdaTAME_ER3_p_file,LambdaTAME_ER3_n_file,t2_color,"LambdaTAME",[(.5, .7),(.4, .5)],"ErdosReyni","--"),
        (LowRankTAME_lrm_ER3_p_file,LowRankTAME_lrm_ER3_n_file,t6_color,"LowRankTAME-lrm",[(.5, .9),(.4, .7)],"ErdosReyni","-."),
        (LowRankTAME_ER3_p_file,LowRankTAME_ER3_n_file,t4_color,"LowRankTAME",[(.5, .98),(.4, .8)],"ErdosReyni",(0, (3, 1, 1, 1)))
    ]
    """
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

    #
    #  Check data
    #
    def check_if_same(file1,file2):
        with open(file1,"r") as f:
            x = json.load(f)

        with open(file2,"r") as f:
            y = json.load(f)

        files_same = False
        for exp_result1,exp_result2 in zip(x,y):
            #check seeds, p_remove, and n are the same
            #(seed,p_remove,n,acc,degree_weighted_acc,matched_tri,A_tri,B_tri,_)
            assert exp_result1[0] == exp_result2[0]
            assert exp_result1[1] == exp_result2[1]
            assert exp_result1[2] == exp_result2[2]
        
            #check A_tris are different
            if exp_result1[-3] == exp_result2[-3]:
                return True
            
            #check B_tris are different
            if exp_result1[-2] == exp_result2[-2]:
                return True
            
        if all([exp_result1[4] ==  exp_result2[4] for (exp_result1,exp_result2) in zip(x,y)]):
            return True
        else:
            return files_same

    """
    ER_path = lambda f: RANDOM_GRAPH_RESULTS + "ERResults/AccuracyExps/" + f
    RG_path = lambda f: RANDOM_GRAPH_RESULTS + "RandomGeometricResults/AccuracyExps/" + f
    for ((p_file1,n_file1,_,_,_,_,_),(p_file2,n_file2,_,_,_,_,_)) in zip(ER3_results,RG_results):
        print(check_if_same(ER_path(p_file1),RG_path(p_file2)))
        print(check_if_same(ER_path(n_file1),RG_path(n_file2)))
    
    for ((p_file1,n_file1,_,_,_,_,_),(p_file2,n_file2,_,_,_,_,_)) in zip(ER3_results,ER_results):
        if check_if_same(ER_path(p_file1),ER_path(p_file2)):
            print(f"same data detected, files:\n{p_file1}\n{p_file2}\n")

        if check_if_same(ER_path(n_file1),ER_path(n_file2)):
            print(f"same data detected, files:\n{n_file1}\n{n_file2}\n")
    """

    #f,axes = plt.subplots(3,2,sharey=True)
    """
    f,axes = plt.subplots(1,3,sharey=True)

    #get simplicial complex differences for experiments

    ER_p_file_edge_diffs = "edge_tri_differences_ER_n:[100]_p_remove[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_pedges:2logn_trialcount:50.json"
    ER2_p_file_edge_diffs = "edge_tri_differences_ER_n:[100]_p_remove[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_pedges:10logn_trialcount:50.json"
    RG_p_file_edge_diffs = "edge_tri_differences_RandomGeometric_degreedist:log5_n:[100]_p_remove[0.0,.002,.004,.006,.008,.01,.015,.02,.04,.06,.1,.2]_trialcount:50.json"
    
    ER_n_file_edge_diffs = "edge_tri_differences_ER_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_pedges:2logn_p_remove:[.01]_trialcount:20.json"
    ER2_n_file_edge_diffs = "edge_tri_differences_ER_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_pedges:10logn_p_remove:[.01]_trialcount:20.json"
    RG_n_file_edge_diffs = "edge_tri_differences_RandomGeometric_degreedist:log5_n:[50,100,150,200,250,300,400,500,600,700,850,1000,1150,1300,1500,1750,2000]_p_remove[.01]_trialcount:20.json"

    def parse_RG_files(file):
        with open(RANDOM_GRAPH_RESULTS + "RandomGeometricResults/AccuracyExps/" + file,"r") as f:
            return json.load(f)

    def parse_ER_files(file):
        with open(RANDOM_GRAPH_RESULTS + "ERResults/AccuracyExps/" + file,"r") as f:
            return json.load(f)
    
    p_sc_diffs = parse_ER_files(ER_p_file_edge_diffs)
    n_sc_diffs = parse_ER_files(ER_n_file_edge_diffs)
    axes[0].set_title("ER p=2log(n)/n")
    for (p_file,n_file,color,annot_label,annotate_locs,graph_type) in ER_results:
        triangle_diff_to_acc(axes[0],p_file,n_file,p_sc_diffs,n_sc_diffs,color,annot_label,annotate_locs,graph_type)

    axes[1].set_title("Random Geometric")
    p_sc_diffs = parse_RG_files(RG_p_file_edge_diffs)
    n_sc_diffs = parse_RG_files(RG_n_file_edge_diffs)
    for (p_file,n_file,color,annot_label,annotate_locs,graph_type) in RG_results:
        triangle_diff_to_acc(axes[1],p_file,n_file,p_sc_diffs,n_sc_diffs,color,annot_label,annotate_locs,graph_type)

    p_sc_diffs = parse_ER_files(ER2_p_file_edge_diffs)
    n_sc_diffs = parse_ER_files(ER2_n_file_edge_diffs)
    axes[2].set_title("ER p=10log(n)/n")
    for (p_file,n_file,color,annot_label,annotate_locs,graph_type) in ER2_results:
        triangle_diff_to_acc(axes[2],p_file,n_file,p_sc_diffs,n_sc_diffs,color,annot_label,annotate_locs,graph_type)
    """


    #f,axes = plt.subplots(3,2,sharey=True)
    f = plt.figure()

    
    # 2 x 3 array 
    height = 0.35
    width = 0.2 
    far_left = .125
    lower_bottom = .125
    upper_bottom = .55
    size_plot_offset = .15
    """
    height = 0.3
    width = 0.4 
    far_left = .1
    lower_bottom = .05
    mid_bottom = .4
    upper_bottom = .
    horizontal_padding = .05
    """
    rectangle1 = [far_left, lower_bottom, width, height]
    rectangle2 = [far_left, upper_bottom, width, height]

    #rectangle1 = [far_left, upper_bottom, width, height]
    #rectangle2 = [far_left + width + horizontal_padding, upper_bottom, width, height]

    ER_noise_ax = [plt.axes(rectangle1),plt.axes(rectangle2)]
    for ax in ER_noise_ax:
        ax.set_yticks([0.0,.25,.5,.75,1.0])


    ER_noise_ax[0].set_xlabel(r"$q$")
    ER_noise_ax[0].set_ylabel("accuracy")
    ER_noise_ax[1].set_ylabel(r"$|T_{M(A,B)}| / \min{\{|T_A|,|T_B|\}}$")
    ER_noise_ax[1].set_title("ER")



   # axes[0,0].set_title("ER p=2log(n)/n")
    #axes[0,0].set_ylabel("Accuracy")


    #axes[1,0].set_xlabel(r"$p_{remove}$")
    #axes[1,0].set_ylabel(r"$\frac{|T_{M(A,B)}|}{\min{|T_A|,|T_B|}}$")
    for (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in ER_results:
        
        TAME_random_graph_noise_experiments(ER_noise_ax, p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle)

    #axes[0,0].set_title("ER p=10log(n)/n")
    #for (p_file,n_file,color,annotate_label,annotate_locs,graph_type) in ER2_results:
    #    TAME_random_graph_noise_experiments(axes[:,0], p_file,n_file,color,annotate_label,annotate_locs,graph_type)
    

    rectangle1 = [far_left+width+.02, lower_bottom, width, height]
    rectangle2 = [far_left+width+.02, upper_bottom, width, height]
    #rectangle1 = [far_left, mid_bottom, width, height]
    #rectangle2 = [far_left + width + horizontal_padding, mid_bottom, width, height]
    
    RG_noise_ax = [plt.axes(rectangle1, sharey=ER_noise_ax[0]),plt.axes(rectangle2, sharey=ER_noise_ax[0])]
    for ax in RG_noise_ax:
        ax.yaxis.set_ticks_position('right')
        xgridlines = ax.get_xgridlines()
        xgridlines[-1].set_color('r')
        xgridlines[-1].set_linestyle((0, (5, 10)))
        ax.yaxis.set_ticks_position('right')

    RG_noise_ax[0].set_xlabel(r"$q$")
    #RG_noise_ax[1].set_title("ER lognormal")
    RG_noise_ax[1].set_title("RG")
    
    #axes[1,1].set_xlabel(r"$p_{remove}$")
    #for (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in ER3_results:
    for (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in RG_results:
        TAME_random_graph_noise_experiments(RG_noise_ax, p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle)


    rectangle1 = [far_left+2*width+size_plot_offset, lower_bottom, width, height]
    rectangle2 = [far_left+2*width+size_plot_offset, upper_bottom, width, height]
    #rectangle1 = [far_left, lower_bottom, width, height]
    #rectangle2 = [far_left + width + horizontal_padding, lower_bottom, width, height]
    
    RG_sizes_ax = [plt.axes(rectangle1, sharey=ER_noise_ax[0]),plt.axes(rectangle2,sharey=ER_noise_ax[0])]
    #axes[2,0].set_title("Random Geometric")
    for ax in RG_sizes_ax:
        ax.yaxis.set_ticks_position('right')

    #RG_sizes_ax[1].set_title("ER lognormal")
    RG_sizes_ax[1].set_title("RG")
    #or (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in ER3_results:
    for (p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle) in RG_results:
        TAME_random_graph_size_experiments(RG_sizes_ax, p_file,n_file,color,annotate_label,annotate_locs,graph_type,linestyle)
    
    #put labels on the plots 
    RG_sizes_labels = [ 
        (t6_color,"LR-TAME-lrm",(.25, .75)),
        (darker_t4_color,"LR-TAME",(.1, .875))
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
    
    """
    #
    #add boxes around experiment groupings
    #
    #Accuracy to Triangle Ratio
    bboxes = [
        mtransforms.Bbox([[0.0, 0.01], [0.335, 0.99]]),
        mtransforms.Bbox([[0.34, 0.01], [0.63, 0.99]])
    ]
    for bb in bboxes:
        p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="Round,pad=0.,rounding_size=.05",
                                ec="b", fc="b",alpha=.1, zorder=0.,
                                )
        f.add_artist(p_bbox)

    #ER to RG graphs
    bboxes = [
        mtransforms.Bbox([[0.0, 0.01], [0.63, 0.49]]),
        mtransforms.Bbox([[0.0, 0.49], [0.63, 0.99]])
    ]
    for bb in bboxes:
        p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="Round,pad=0.,rounding_size=.05",
                                ec="g", fc="g",alpha=.1, zorder=0.,
                                )
        f.add_artist(p_bbox)

    #noise to size experiments 
    bb = mtransforms.Bbox([[0.34, 0.01], [0.95, 0.99]])
    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                            abs(bb.width), abs(bb.height),
                            boxstyle="Round,pad=0.,rounding_size=.05",
                            ec="r", fc="r",alpha=.1, zorder=0.,
                            )
    f.add_artist(p_bbox)
    """
    plt.show()

def triangle_diff_to_acc(axes,p_file,n_file,p_sc_diffs,n_sc_diffs,color,anotate_label,anotate_loc,graph_type):
    #
    #  Load the files
    #

    if graph_type == "RandomGeometric":
        def load_data(files):
            data = []
            for file in files:
                with open(RANDOM_GRAPH_RESULTS + "RandomGeometricResults/AccuracyExps/" + file,"r") as f:
                    data.extend(json.load(f))
            return data
    elif graph_type == "ErdosReyni":
        def load_data(files):
            data = []
            for file in files:
                with open(RANDOM_GRAPH_RESULTS + "ERResults/AccuracyExps/" + file,"r") as f:
                    data.extend(json.load(f))
            return data
    else: 
        raise ValueError(f"graph type:{graph_type} must be one of 'RandomGeometric' or 'ErdosReyni'")

    p_data = load_data([p_file])
    n_data = load_data([n_file])


    #
    #  Parse the data
    #

    def process_data(data,simplicial_complex_diffs):

        acc_data = []
        edge_diffs = []
        tri_diffs = []

        for (exp_results,sc_diffs) in zip(data,simplicial_complex_diffs):
            (seed,p_remove,n,acc,degree_weighted_acc,matched_tri,A_tri,B_tri,_) = exp_results
            (sc_seed,sc_p_remove,sc_n,_,edge_diff,tri_diff) = sc_diffs
            assert (sc_seed == seed)
            assert (sc_p_remove == p_remove)
            assert (sc_n == n) 

            acc_data.append(acc)
            edge_diffs.append(edge_diff)
            tri_diffs.append(tri_diff)

        return  acc_data, edge_diffs , tri_diffs 

    tri_to_acc_data, edge_diffs ,tri_diffs = process_data(p_data,p_sc_diffs)


    n_tri_to_acc_data, n_edge_diffs , n_tri_diffs = process_data(n_data,n_sc_diffs)
    tri_to_acc_data.extend(n_tri_to_acc_data)
    edge_diffs.extend(n_edge_diffs)
    tri_diffs.extend(n_tri_diffs)

    print(max(edge_diffs))
    print(max(tri_diffs))
  

    #
    #  Tri to Acc plots
    #
    ax = axes
    #ax = axes[1,0]

    ax.set_ylim(0,1.0)
    ax.set_yticks([0.0,.25,.5,.75,1.0])

    #ax.set_xlim(1,10000)
    #ax.set_xticks([0.0,.01,.02,.03,.04,.05])
    #ax.set_xscale("log")

    ax.set_xlabel(r"$\frac{1}{6}\|T(A) - T(B)\|_1$")
    ax.set_ylabel("accuracy")
    ax.grid(which="major", axis="both",alpha=.2)

    #xgridlines = ax.get_xgridlines()
    #xgridlines[1].set_color('r')

    def make_scatter_plot(plot_ax,domain,data):

        #tri_diff = [x[0] for x in data]
        #accuracies = [x[1] for x in data]

        #plot_ax.scatter(tri_diff,accuracies,c=color,s=5,alpha=.5)
        min_x = min(domain)
        max_x = max(domain)
        scaled_domain = [(x - min_x)/max_x for x in domain]
        xout,yout = plot_1d_loess_smoothing(scaled_domain,data,.3)

        ax.plot([x*max_x + min_x for x in xout],yout,c=color,linestyle="-")
        ax.annotate(anotate_label,xy=anotate_loc[0], xycoords='axes fraction', c=color)


    make_scatter_plot(ax,tri_diffs,tri_to_acc_data)


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
 #   ax.set_yscale("lo")

    def make_percentile_plot(plot_ax, data,color,**kwargs):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (30,70,.2,color)
        ]

        sizes = list(data.keys())
        perm = sorted(range(len(sizes)),key=lambda i:sizes[i])
        sizes = np.array([sizes[i] for i in perm])

        x = list(data.values())
        size_exp_data = np.array([x[i] for i in perm]).T

        plot_percentiles(plot_ax,  size_exp_data, sizes, lines, ribbons,**kwargs)


    #make_percentile_plot(ax,{k:v for k,v in n_acc.items() if k == .01},color)
    make_percentile_plot(ax,n_acc,color,linestyle=linestyle)
    #ax.annotate(anotate_label,xy=anotate_loc[1], xycoords='axes fraction', c=color)
    
    # ---------------  Degree Weighted Accuracies   ---------------- #
    """
    ax = axes[1,1]
    ax.set_xlim(50,2000)
    ax.set_yticks([0.0,.25,.5,.75,1.0])
    ax.set_xticks([100,500,1000,2000])
    ax.set_xlabel(r"n")
    ax.grid(which="major", axis="both",alpha=.2)
    ax.yaxis.set_ticks_position('right')
    #ax.spines['left'].set_color('r')
 #   ax.set_yscale("lo")

    make_percentile_plot(ax,n_dw_acc,color)
    """
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
    #ax.set_xlim(1e-2,.05)
    #ax.set_xticks([0.0,.01,.02,.03,.04,.05])
    ax.set_xscale("log")
    #ax.set_xlabel(r"$p_{remove}$")
    #ax.set_ylabel("accuracy")
    ax.grid(which="major", axis="both",alpha=.3)
    #xgridlines = ax.get_xgridlines()
    #xgridlines[1].set_color('r')
    
    def make_percentile_plot(plot_ax, data,color,linestyle):
        #TODO: check on scope priorities of ax for closures   
        lines = [(lambda col: np.percentile(col,50),1.0,color) ]
        ribbons = [
            (30,70,.2,color)
        ]

        noise_levels = list(data.keys())
        perm = sorted(range(len(noise_levels)),key=lambda i:noise_levels[i])
        noise_levels = np.array([noise_levels[i] for i in perm])

        data = list(data.values())
        data = np.array([data[i] for i in perm]).T

        plot_percentiles(plot_ax,  data, noise_levels, lines,ribbons,linestyle=linestyle)

    make_percentile_plot(ax,p_acc,color,linestyle=linestyle)

    # ----------------  Degree Weighted Accuracies   ---------------------#
    """ 
    ax = axes[1,0]

    ax.set_ylim(0,1.0)
    #ax.set_xlim(1e-2,.05)
    #ax.set_xticks([0.0,.01,.02,.03,.04,.05])
    ax.set_xscale("log")
    #ax.set_yticks([0.0,.25,.5,.75,1.0])
    ax.set_xlabel(r"$p_{remove}$")
    ax.set_ylabel("d.w. accuracy")
    ax.grid(which="major", axis="both",alpha=.2)

    make_percentile_plot(ax,p_dw_acc,color)
    """
    # ------------------------  Tri Match Ratios   ---------------------#
    
    ax = axes[1]
    ax.set_ylim(0,1.0)
    #ax.set_xlim(1e-2,.05)
    #ax.set_xticks([0.0,.01,.02,.03,.04,.05])
    ax.set_xscale("log")
    #ax.set_yticks([0.0,.25,.5,.75,1.0])
    #ax.set_xlabel(r"$p_{remove}$")
    #ax.set_ylabel(r"$\frac{|T_{M(A,B)}|}{min(|T_A|,|T_B|)}$")
    ax.grid(which="major", axis="both",alpha=.3)

    make_percentile_plot(ax,p_tri_match,color,linestyle=linestyle)

    #ax.annotate(anotate_label,xy=anotate_loc[0], xycoords='axes fraction', c=color)
    