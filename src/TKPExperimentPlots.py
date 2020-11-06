import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json
import itertools


#from scipy.stats import percentileofscore
from loess.loess_1d import loess_1d
from functools import reduce
from math import floor,ceil
import seaborn as sns

import matplotlib.ticker as mticker #used to plot scatter 3d in log scale
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.transforms as mtransforms
#import matplotlib.patches as mpatch
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LogFormatter
#from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

#
#   Code is expected to be run in the src directory
#


#PROJECT_LOCATION = "/Users/ccolley/PycharmProjects/PAMI_TAME_Exps/"
#MULTIMAGNA_RANK_RESULTS = PROJECT_LOCATION+"data/MultiMAGNA_TAME_results/RankExps/"
#MULTIMAGNA_SING_VALS_RESULTS = PROJECT_LOCATION+"data/MultiMAGNA_TAME_results/SingVals/"
HOFASM_RESULTS = "../data/HOFASMexperiments/"
#random graph results locations
#RANDOM_GRAPH_RESULTS = PROJECT_LOCATION + "data/synthetic_TAME_results/"

MY_DPI = 96  #used to convert into pixels
#colors

t1_color = '#fb6b5b'  # red
darker_t1_color = "#E34635"
darkest_t1_color = "#BB2617"

t4_color = '#7ecbe0'  # blue
darker_t4_color = "#56AFC9"
darkest_t4_color = "#3590AA"

t3_color = "#aa42a3" #purple

t2_color = "#00a878" #green
darkest_t2_color = "#00664A"
lightest_t2_color = "#A7EBD8"

t5_color = "#fbc04c" #yellow
t6_color = "b"
#t6_color = "#ff7f50" #coral
