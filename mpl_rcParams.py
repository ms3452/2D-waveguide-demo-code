import matplotlib.gridspec as gridspec
import matplotlib as mpl

rcparams = {
    # Use LaTeX to write all text
    "text.usetex": False, 
    "font.size": 7,
    "legend.fontsize": 6,
    'legend.title_fontsize': 6,
    "lines.linewidth": 0.7,
    "lines.markersize": 6,
    "mathtext.fontset" : "custom",
    "mathtext.rm" : "sans",
    "mathtext.default" : "regular",
    "figure.autolayout": False ,
    "figure.dpi" : 150,
    "figure.figsize" : [2.5,1.8],
    "axes.grid" : False,
    "axes.linewidth" : 0.3,
    'axes.spines.right': True,
    'axes.spines.top': True,
    "axes.labelsize": 6,
    "axes.axisbelow" : False,
    'axes.formatter.use_mathtext': True,
    "xtick.labelsize": 6,
    "xtick.direction" : "out",
    "xtick.bottom" : True,
    'xtick.major.size': 2,
    'xtick.major.width': 0.3,
    'xtick.minor.width': 0.2,
    'xtick.alignment': 'center',
    'ytick.major.size': 2,
    'ytick.major.width': 0.3,
    'ytick.minor.width': 0.2,
    "ytick.direction" : "out",
    "ytick.left" : True,
    "ytick.labelsize": 6,
    "legend.frameon" : False,
    "patch.facecolor": 'white',
    "axes.labelsize": 6.5,  # Updated font size for x and y labels
}

mpl.rcParams.update(rcparams)