###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import cm



def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting
    for i, feature in enumerate(['capital-gain','capital-loss']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Skewed Distributions of Continuous Census Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 4, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Set additional plots invisibles
    ax[0, 3].set_visible(False)
    ax[1, 3].axis('off')

    # Create legend
    for i, learner in enumerate(results.keys()):
        pl.bar(0, 0, color=colors[i], label=learner)
    pl.legend()
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
    

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  

def model_picking(clfs,df,selection):
    graphs = {'acc_test' : ['Accuracy Testing Set',None],
              'acc_train' : ['Accuracy Training Set',None],
              'f_test' : [r'$F_{0.5}$ Score Testing Set',None],
              'f_train' : [r'$F_{0.5}$ Score Training Set',None],
              'pred_time' : [r'Prediction Time (s)',None],
              'train_time' : [r'Training Time (s)',None],
              'acc_dif' : ['Accuracy', '$\Delta(Training,Testing)$'],
              'f_dif' : ['$F_{0.5}$ Score', '$\Delta(Training,Testing)$']}

    pl.gcf().clear()
    fig = pl.figure(figsize = (13,8))
    bar_width = 0.085
    colors = cm.hsv(np.linspace(1,255,11).astype(int))

    for k, key, in enumerate(['acc', 'f']):
        ax = fig.add_subplot(2,2,k+1)
        count = 0
        for l, clf, in enumerate(clfs):
            for i in range(0,3):
                if (i == 2)&(k == 1):
                    ax.bar(i+l*bar_width,
                           df[key+'_test'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)],
                           width = bar_width,
                           color = colors[l],
                           alpha=0.2)
                    if l == len(clfs)-1:
                        ax.annotate('Testing',
                                    xy=(i+l*bar_width,df[key+'_test'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)]),
                                    xycoords = 'data',
                                    xytext =((i+l*bar_width)*1.15,df[key+'_test'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)]),
                                    arrowprops = {'arrowstyle':'->'},
                                    fontsize=14)

                    ax.bar(i+l*bar_width,
                       df[key+'_dif'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)],
                       width = bar_width,
                       color = colors[l],
                       bottom = df[key+'_test'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)],
                       label=clf.__class__.__name__,
                       alpha=selection[clf.__class__.__name__][0])

                    if l == len(clfs)-1:
                        ax.annotate('Training',
                                    xy=(i+l*bar_width,df[key+'_train'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)]),
                                    xycoords = 'data',
                                    xytext =((i+l*bar_width)*1.15,df[key+'_train'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)]),
                                    arrowprops = {'arrowstyle':'->'},
                                    fontsize=14)

                    ax.legend(bbox_to_anchor=(1.00, 0.15),
                              fontsize =14,
                              frameon=False,
                              labelspacing=0.5)
                else:
                    ax.bar(i+l*bar_width,
                           df[key+'_test'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)],
                           width = bar_width,
                           color = colors[l],
                           alpha=0.2)
                    ax.bar(i+l*bar_width,
                           df[key+'_dif'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)],
                           width = bar_width,
                           color = colors[l],
                           bottom = df[key+'_test'][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)],
                           alpha=selection[clf.__class__.__name__][0])

        ax.set_ylabel(graphs[key+'_dif'][1],fontsize=14)
        ax.set_title(graphs[key+'_dif'][0],fontsize=14)
        ax.set_ylim(df[key+'_test'][df[key+'_test']!=0].min()*0.95,df[key+'_train'].max()*1.05)
        ax.tick_params(axis='x',
                   which='both',
                   bottom=False,
                   top=False,
                   labelbottom=False)


    tick = np.zeros([2,30])
    tick[1,:] = np.repeat(np.array([0,1,2]),10)
    for k, key in enumerate(['pred_time', 'train_time']):
        ax = fig.add_subplot(2,2,k+3)
        count = 0
        for i in range(0,3):
            for l, clf, in enumerate(clfs):
                ax.bar(i+l*bar_width,
                       df[key][(df['clf']==clf.__class__.__name__)&(df['sample_size']==i)],
                       width = bar_width,
                       color = colors[l],
                       alpha=selection[clf.__class__.__name__][0])
                tick[0,count] = i+l*bar_width
                count += 1
            tick[0,:][tick[1,:]==i] = np.mean(tick[0,:][tick[1,:]==i])

        ax.set_xticks(np.unique(np.sort(tick[0,:])))
        ax.set_xticklabels(['1%','10%','100%'],fontsize=14)

        ax.set_yscale('log')
        ax.set_xlabel('Sample Size \n (%Training Set)',fontsize = 14)
        ax.set_title(graphs[key][0],fontsize = 14)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)

#BIAS,JUST GOOD, VARIANCE    
    fig = pl.figure(figsize = (13,10))
    for e, element ,in enumerate(['acc','f']):
        ax = fig.add_subplot(2,1,e+1)
        for l, clf, in enumerate(clfs):
            ax.plot(df['sample_size'][df['clf']==clf.__class__.__name__],
                    df[element+'_train'][df['clf']==clf.__class__.__name__],
                   color=colors[l],
                   label=clf.__class__.__name__ + '_train',
                   alpha=selection[clf.__class__.__name__][0])
            ax.plot(df['sample_size'][df['clf']==clf.__class__.__name__],
                    df[element+'_test'][df['clf']==clf.__class__.__name__],
                    '-.',
                    color=colors[l],
                   label=clf.__class__.__name__ + '_test',
                   alpha=selection[clf.__class__.__name__][0])

        if element == 'acc':
            ax.set_ylabel('Accuracy',fontsize=14)
            ax.legend(bbox_to_anchor=(1.00, 1.00))
            ax.tick_params(axis='x',
                          which='both',
                          bottom=False,
                          top=False,
                          labelbottom=False)
        else:
            ax.set_ylabel(r'$F_{0.5}\,\,Score$',fontsize=14)
            ax.set_ylim(0.5,1.05)
            ax.set_xticks([0,1,2])
            ax.set_xticklabels(['1%','10%','100%'],fontsize=14)
            ax.set_xlabel('Sample Size \n (%Training Set)',fontsize=14)

    fig.subplots_adjust(hspace=0.1)