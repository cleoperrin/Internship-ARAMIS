import os
import numpy as np
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt

from scipy.io import loadmat
from scipy import stats
from mne.stats import bonferroni_correction, fdr_correction


# FOR A FIRST APPROACH AND VISUALISATION

# Function for visual comparison between p-values between conditions, obtained with the 2 toolboxes for a given subject
def plot_ttest_con_tbx(pvaluesMNE, pvaluesBST, name_channels, title) :
    """
    Take p-values of statistical test conducted on MNE and BST, as inputs and plot both p-values : MNE (left)
    and BST (right). Can be corrected or not, depending on the user input.
    ## inputs:
        # pvaluesMNE: generally of the form "t_test_MNE_1.pvalue", p-values of the test conducted between conditions
                      on MNE connectivity results.
        # pvaluesBST: p-values of the test conducted between conditions on BST connectivity results
        # name_channels: list of channel labels to be axis labels of the heat map
        # title : title that will be given to the graph
    ## output:
        # A plot made of two subplots of a p-value heatmaps for MNE (left) and BST (right)
    """
    answer = input('Would you like the p-values to be corrected before plotted ? [y/n].\n')
    if answer == 'y':
        answer_meth = input('Which correction ? [bonferroni / fdr].\n')
        if answer_meth == 'bonferroni' or 'fdr' :
            from Correction import pval_correction
            pvaluesMNE = pval_correction(pvaluesMNE, answer_meth)
            pvaluesBST = pval_correction(pvaluesBST, answer_meth)
        else :
                print('Correction method not known.')
    elif answer == 'n':
        pvaluesMNE = pvaluesMNE
        pvaluesBST = pvaluesBST

    mask = np.triu(np.ones_like(pvaluesMNE, dtype=bool))
    fig, (ax0,ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.heatmap(pvaluesMNE,
                ax=ax0,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="RdPu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax0.set_title('MNE',
                  fontstyle='italic', fontweight='bold', fontsize=15)
    sns.heatmap(pvaluesBST,
                ax=ax1,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="RdPu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax1.set_title('Brainstorm',
                  fontstyle = 'italic', fontweight='bold', fontsize=15)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    ax1.set_xticklabels(name_channels, rotation=90)
    ax0.set_xticklabels(name_channels, rotation=90)
    ax1.set_yticklabels(name_channels)
    plt.show()

# Plotting p-values of wilcoxon test between conditions for each toolboxes, and return channel pairs for which
# the difference of connectivity between left and right conditions is the most significant. These results can be
# used in rainplot_channel() to focus on the most important channels with regards to the research question
def plot_wil_con_tbx(pvaluesMNE, pvaluesBST, name_channels, title):
    """
    Take p-values of wilcoxon test conducted between conditions for the 2 toolboxes at the group level as inputs
    and plot both p-values as heat-map. Can be corrected or not, depending on the user input. Depending on user input
    also returns for each toolboxes the channel pairs with the lowest p-value, thus the one for which the difference
    of connectivity between conditions is the most significant.
    ## inputs:
        # pvaluesMNE: generally of the form "results_wil_test_MNE", p-values of the wilcoxon test conducted
                    between conditions with MNE connectivity results
        # pvaluesBST: p-values of the wilcoxon test conducted between conditions with BST connectivity results
        # name_channels: list of channel labels to be axis labels of the heat map
        # title : title that will be given to the graph
    ## output:
        # List of channel pairs which FC values are highly discriminant between conditions
        # A plot made of two subplots of a p-value heatmaps for MNE wilcoxon between conditions (left)
        and BST wilcoxon between conditions (right)
    """
    # Return firslty the lowest pvalue of the array : the channel pair for which the difference is more important
    # Remove NaN, and apply np.amin to find the lowest p-value in the array
    min_pval_MNE = np.amin(pvaluesMNE[~np.eye(pvaluesMNE.shape[0], dtype=bool)].reshape(pvaluesMNE.shape[0], -1))
    min_pval_BST = np.amin(pvaluesBST[~np.eye(pvaluesBST.shape[0], dtype=bool)].reshape(pvaluesBST.shape[0], -1))
    # Get the indices of that value (only on the lower part of matrix, as symmetric)
    indices_min_MNE = np.argwhere(pvaluesMNE == min_pval_MNE)[-1]
    indices_min_BST = np.argwhere(pvaluesBST == min_pval_BST)[-1]
    indices_all_min_MNE = np.argwhere(pvaluesMNE == min_pval_MNE)[-(int(len(np.argwhere(pvaluesMNE == min_pval_MNE))/2)):]
    indices_all_min_BST = np.argwhere(pvaluesBST == min_pval_BST)[-(int(len(np.argwhere(pvaluesBST == min_pval_BST))/2)):]
    # Get the name of the channels associated
    channel_1_MNE, channel_2_MNE = name_channels[indices_min_MNE[0]], name_channels[indices_min_MNE[1]]
    channel_1_BST, channel_2_BST = name_channels[indices_min_BST[0]], name_channels[indices_min_BST[1]]

    print('For MNE, the most discriminant channel pair between conditions is : ', [channel_1_MNE, channel_2_MNE],
          ', with indices : ', indices_min_MNE, '\n (p-value of : ',min_pval_MNE,'). Also applies for ',
          int(len(indices_all_min_MNE)-1),' other channel pairs.')
    if len(indices_all_min_MNE > 1) :
        answer = input('Would you like to see the other channel pairs concerned by the lowest p-value ? [y/n].\n')
        if answer == 'y':
            liste_channels = []
            for i in range(len(indices_all_min_MNE)-1):
                liste_channels.append([name_channels[indices_all_min_MNE[i][0]], name_channels[indices_all_min_MNE[i][1]]])
            print('\n', liste_channels, '\n')
        elif answer == 'n':
            print('Done')
        else :
            print('You should answer with y or n.')

    print('For BST, the most discriminant channel pair between conditions is : ', [channel_1_BST, channel_2_BST],
          ', with indices : ', indices_min_BST, '\n (p-value of : ',min_pval_BST,'). Also applies for ',
          int(len(indices_all_min_BST)-1),' other channel pairs.')
    if len(indices_all_min_BST > 1) :
        answer = input('Would you like to see the other channel pairs concerned by the lowest p-value ? [y/n].\n')
        if answer == 'y':
            liste_channels = []
            for i in range(len(indices_all_min_BST)-1):
                liste_channels.append([name_channels[indices_all_min_BST[i][0]], name_channels[indices_all_min_BST[i][1]]])
            print('\n', liste_channels, '\n')
        elif answer == 'n':
            print('Done')
        else:
            print('You should answer with y or n.')

    if len(indices_all_min_MNE > 1) and len(indices_all_min_BST > 1):
        answer = input('Would you like to see the channel pairs having the smallest p-value in BOTH toolboxes ? [y/n].\n')
        if answer == 'y':
            liste_channels_MNE = []
            for i in range(len(indices_all_min_MNE) - 1):
                liste_channels_MNE.append([name_channels[indices_all_min_MNE[i][0]], name_channels[indices_all_min_MNE[i][1]]])
            liste_channels_BST = []
            for i in range(len(indices_all_min_BST) - 1):
                liste_channels_BST.append([name_channels[indices_all_min_BST[i][0]], name_channels[indices_all_min_BST[i][1]]])
            tupple_channels_MNE, tupple_channels_BST = set(map(tuple, liste_channels_MNE)), set(map(tuple, liste_channels_BST))
            liste_common = list(set(tupple_channels_MNE).intersection(tupple_channels_BST))
            print('\n', liste_common, '\n')
        elif answer == 'n':
            print('Done')
        else:
            print('You should answer with y or n.')

    # Plot of pvalues
    answer_corr = input('\n Would you like the p-values to be corrected before plotted ? [y/n].\n')
    if answer_corr == 'y':
        answer_meth = input('Which correction ? [bonferroni / fdr].\n')
        if answer_meth == 'bonferroni' or 'fdr' :
            from Correction import pval_correction
            pvaluesMNE = pval_correction(pvaluesMNE, answer_meth)
            pvaluesBST = pval_correction(pvaluesBST, answer_meth)
        else :
                print('Correction method not known.')
    elif answer_corr == 'n':
        pvaluesMNE = pvaluesMNE
        pvaluesBST = pvaluesBST

    mask = np.triu(np.ones_like(pvaluesMNE, dtype=bool))
    fig, (ax0,ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.heatmap(pvaluesMNE,
                ax=ax0,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="RdPu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax0.set_title('MNE',
                  fontstyle='italic', fontweight='bold', fontsize=15)
    sns.heatmap(pvaluesBST,
                ax=ax1,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="RdPu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax1.set_title('Brainstorm',
                  fontstyle = 'italic', fontweight='bold', fontsize=15)
    fig.suptitle(title,fontsize = 15, fontweight= 'bold')
    ax1.set_xticklabels(name_channels, rotation=90)
    ax0.set_xticklabels(name_channels, rotation=90)
    ax1.set_yticklabels(name_channels)
    plt.show()


# FOR A GROUP COMPARISON

# Plotting p-values of wilcoxon test between toolboxes for both conditions
def plot_group_wil_tbx(pvaluesL, pvaluesR, name_channels, title):
    """
    Take p-values of wilcoxon test conducted between toolboxes and for the 2 conditions, as inputs
    and plot both p-values as heat-map. Can be corrected or not, depending on the user input.
    ## inputs:
        # pvaluesL: generally of the form "results_wil_test_L", p-values of the test conducted on left condition
                    connectivity results
        # pvaluesR: p-values of the test conducted on right condition connectivity results
        # name_channels: list of channel labels to be axis labels of the heat map
        # title : title that will be given to the graph
    ## output:
        # A plot made of two subplots of a p-value heatmaps for left condition (left) and right condition (right)
    """

    answer = input('Would you like the p-values to be corrected before plotted ? [y/n].\n')
    if answer == 'y':
        answer_meth = input('Which correction ? [bonferroni / fdr].\n')
        if answer_meth == 'bonferroni' or 'fdr' :
            from Correction import pval_correction
            pvaluesL = pval_correction(pvaluesL, answer_meth)
            pvaluesR= pval_correction(pvaluesR, answer_meth)
        else :
                print('Correction method not known.')
    elif answer == 'n':
        pvaluesL = pvaluesL
        pvaluesR = pvaluesR

    mask = np.triu(np.ones_like(pvaluesL, dtype=bool))
    fig, (ax0,ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.heatmap(pvaluesL,
                ax=ax0,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="GnBu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax0.set_title('Left condition',
                  fontstyle='italic', fontweight='bold', fontsize=25)
    sns.heatmap(pvaluesR,
                ax=ax1,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="GnBu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax1.set_title('Right condition',
                  fontstyle = 'italic', fontweight='bold', fontsize=25)
    fig.suptitle(title,fontsize = 15, fontweight= 'bold')
    ax1.set_xticklabels(name_channels, rotation=90, fontsize = 15)
    ax0.set_xticklabels(name_channels, rotation=90, fontsize = 15)
    ax0.set_yticklabels(name_channels, fontsize=15)
    plt.show()


# BASED ON THESE FIRST RESULTS, CONDUCT FURTHER ANALYSIS AT THE INDIVIDUAL SCALE

# Comparing the difference of connectivity values across sensors for a given subject
def diff_conn(dataMNE_L, dataMNE_R, dataBST_L, dataBST_R, name_channels, title) :
    """
    Take connectivity values obtained from MNE and BST for one subject and the two conditions
    as inputs, compute the mean across trials for both toolboxes and plot the difference (MNE - BST)
    between connectivity results obtained from the two toolboxes
    ## inputs:
        # dataMNE_L: generally of the form "MNE_subject_1_L", connectivity values for a specific subject and left
                    condition obtained from MNE
        # dataMNE_R: connectivity values for a specific subject and right condition obtained from MNE
        # dataBST_L: connectivity values for a specific subject and left condition obtained from BST
        # dataBST_R: connectivity values for a specific subject and right condition obtained from BST
        # name_channels: list of channel labels to be axis labels of the heat map
        # title : title that will be given to the graph
    ## output:
        # A difference heatmap in all-sensor space between connectivity results of MNE and BST
    """
    mean_BST_L, mean_MNE_L = np.mean(dataBST_L, axis=0), np.mean(dataMNE_L, axis=0)
    mean_BST_R, mean_MNE_R = np.mean(dataBST_R, axis=0), np.mean(dataMNE_R, axis=0)
    diff_L = mean_MNE_L - mean_BST_L
    diff_R = mean_MNE_R - mean_BST_R

    mask = np.triu(np.ones_like(diff_L, dtype=bool))
    fig, (ax0,ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.heatmap(diff_L,
                ax=ax0,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="bwr",
                square=True,
                vmin=-0.02, vmax=0.02,
                mask=mask,
                cbar_kws={'label': 'Coherence value difference'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax0.set_title('Left condition',
                  fontstyle='italic')
    sns.heatmap(diff_R,
                ax=ax1,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="bwr",
                square=True,
                vmin=-0.02, vmax=0.02,
                mask=mask,
                cbar_kws={'label': 'Coherence value difference'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax1.set_title('Right condition',
                  fontstyle = 'italic')
    fig.suptitle(title, fontsize=15, fontweight='bold')
    ax1.set_xticklabels(name_channels, rotation=90)
    ax0.set_xticklabels(name_channels, rotation=90)
    ax1.set_yticklabels(name_channels)
    plt.show()

# Compute and plot results t-test of connectivity obtained between the 2 toolboxes for a given subject
def pvalues_ttest_tbx(conn_MNE_L, conn_MNE_R, conn_BST_L, conn_BST_R, name_channels, title) :
    """
    Take connectivity values obtained from MNE and BST for one subject and both conditions
    as inputs, conduct the t-test between these values and return a heat-map of the p-values.
    Can be corrected or not, depending on the user input.
    ## inputs:
        # conn_MNE_L: generally of the form "MNE_subject_1_L", connectivity values for a specific subject
                    and left condition obtained from MNE
        # conn_MNE_R: connectivity values for a specific subject and right condition obtained from MNE
        # conn_BST_L: connectivity values for a specific subject and left condition obtained from BST
        # conn_BST_R: connectivity values for a specific subject and right condition obtained from BST
        # name_channels: list of channel labels to be axis labels of the heat map
        # title : title that will be given to the graph
    ## output:
        # A p-value heatmap in all-sensor space between connectivity results of MNE and BST
    """

    t_test_toolboxes_L = (stats.ttest_rel(conn_MNE_L, conn_BST_L)).pvalue
    t_test_toolboxes_R = (stats.ttest_rel(conn_MNE_R, conn_BST_R)).pvalue

    answer = input('Would you like the p-values to be corrected before plotted ? [y/n].\n')
    if answer == 'y':
        answer_meth = input('Which correction ? [bonferroni / fdr].\n')
        if answer_meth == 'bonferroni' or 'fdr' :
            from Correction import pval_correction
            t_test_toolboxes_L = pval_correction(t_test_toolboxes_L, answer_meth)
            t_test_toolboxes_R = pval_correction(t_test_toolboxes_R, answer_meth)
        else :
                print('Correction method not known.')
    elif answer == 'n':
        t_test_toolboxes_L = t_test_toolboxes_L
        t_test_toolboxes_R = t_test_toolboxes_R

    mask = np.triu(np.ones_like(t_test_toolboxes_L, dtype=bool))
    fig, (ax0,ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.heatmap(t_test_toolboxes_L,
                ax=ax0,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="GnBu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax0.set_title('Left condition',
                  fontstyle='italic', fontweight='bold')
    sns.heatmap(t_test_toolboxes_R,
                ax=ax1,
                cbar=True,
                cbar_ax=cbar_ax,
                cmap="GnBu_r",
                square=True,
                vmin=0, vmax=0.05,
                mask=mask,
                cbar_kws={'label': 'p-value'},
                xticklabels=True, yticklabels=True,
                linewidths=.5)
    ax1.set_title('Right condition',
                  fontstyle = 'italic', fontweight='bold')
    fig.suptitle(title, fontsize=15, fontweight='bold')
    ax1.set_xticklabels(name_channels, rotation=90)
    ax0.set_xticklabels(name_channels, rotation=90)
    ax1.set_yticklabels(name_channels)
    plt.show()

# Comparing the connectivity values as rainplot for given subject (average across connectivity values of channel pairs)
def rainplot_tbx_cond(dataMNE_L, dataMNE_R, dataBST_L, dataBST_R, title):
    """
    Take connectivity values obtained from MNE and BST and for the 2 conditions for one subject
    as inputs, returns a rainplot of connectivity values for both conditions and toolboxes.
    Coherence values are obtained by averaging across channel pairs and trials (144 points, one for each trial).
    ## inputs:
        # dataMNE_L: generally of the form "MNE_subject_1_L", connectivity values for a specific subject
                    and left condition obtained from MNE
        # dataMNE_R: generally of the form "MNE_subject_1_R", connectivity values for a specific subject
                    and right condition obtained from MNE
        # dataBST_L: connectivity values for a specific subject and left condition obtained from BST
        # dataBST_R: connectivity values for a specific subject and right condition obtained from BST
        # title : title that will be given to the graph
    ## output:
        # Four rainplots of coherence values across conditions (left and right) and toolboxes (BST and MNE) for a
        given subject by averaging across channel pairs values
    """

    data_R = []
    data_L = []
    for i in range(144):
        data_L.append(np.mean(dataMNE_L[i]))  # Average across channels
        data_R.append(np.mean(dataMNE_R[i]))
    for i in range(144):
        data_L.append(np.mean(dataBST_L[i]))
        data_R.append(np.mean(dataBST_R[i]))
    df = pd.DataFrame(data={'Toolbox': ['MNE'] * 144 + ['BST'] * 144, 'Left condition': data_L, 'Right condition': data_R})

    f, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    f.suptitle(title, fontweight='bold', fontsize=15)

    pt.half_violinplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, palette='pastel', bw=.2, cut=0.,
                                 scale="area", width=.6, inner=None, orient="h")
    sns.stripplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, palette='pastel', edgecolor="white",
                            size=3, jitter=1, zorder=0, orient="h")
    sns.boxplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, color="black", width=.15, zorder=10,
                          showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                          showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                          saturation=1, orient="h")
    axes[0].set_title('Left condition', fontstyle='italic', fontweight = 'bold')
    axes[0].set_yticklabels(['MNE', 'Brainstorm'])
    axes[0].set_xlabel('', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('', fontweight='bold', fontsize=12)

    pt.half_violinplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, palette='pastel', bw=.2, cut=0.,
                                 scale="area", width=.6, inner=None, orient="h")
    sns.stripplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, palette='pastel', edgecolor="white",
                            size=3, jitter=1, zorder=0, orient="h")
    sns.boxplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, color="black", width=.15, zorder=10,
                          showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                          showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                          saturation=1, orient="h")
    axes[1].set_title('Right condition', fontstyle='italic', fontweight = 'bold')
    axes[1].set_yticklabels(['MNE', 'Brainstorm'])
    axes[1].set_xlabel('Coherence values', fontsize=10)
    axes[1].set_ylabel('', fontweight='bold', fontsize=12)


# Rain plot of connectivity values for 1 channel pair and 1 subject for each conditions and toolboxes
def rainplot_channel(dataMNE_L, dataMNE_R, dataBST_L, dataBST_R, name_channels, channel1, channel2, title):
    """
    Take connectivity values obtained from MNE and BST and for the 2 conditions for one subject
    as inputs, returns a rainplot of connectivity values for both conditions and toolboxes.
    The connectivity values considered are for 1 channel pair only, specified with its names, and each plot
    therefore comprises 144 points, one for each trial.
    ## inputs:
        # dataMNE_L: generally of the form "MNE_subject_1_L", connectivity values for a specific subject
                    and left condition obtained from MNE
        # dataMNE_R: generally of the form "MNE_subject_1_R", connectivity values for a specific subject
                    and right condition obtained from MNE
        # dataBST_L: connectivity values for a specific subject and left condition obtained from BST
        # dataBST_R: connectivity values for a specific subject and right condition obtained from BST
        # name_channels: list of channel labels
        # channel1: name of first channel composing channel pair interested in
        # channel2: name of second channel composing channel pair interested in
        # title : title that will be given to the graph
    ## output:
        # Four rainplots of coherence values across conditions (left and right) and toolboxes (BST and MNE) for a
        given subject and given channel pair
    """
    coorx = name_channels.index(channel1)
    coory = name_channels.index(channel2)
    data_R = []
    data_L = []
    for i in range(144):
        data_L.append(np.mean(dataMNE_L[i][coorx][coory]))  # Average across channels
        data_R.append(np.mean(dataMNE_R[i][coorx][coory]))
    for i in range(144):
        data_L.append(np.mean(dataBST_L[i][coorx][coory]))
        data_R.append(np.mean(dataBST_R[i][coorx][coory]))
    df = pd.DataFrame(data={'Toolbox': ['MNE'] * 144 + ['BST'] * 144, 'Left condition': data_L, 'Right condition': data_R})


    f, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    f.suptitle(title, fontweight='bold', fontsize=15)

    pt.half_violinplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, palette='pastel', bw=.2, cut=0.,
                                 scale="area", width=.6, inner=None, orient="h")
    sns.stripplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, palette='pastel', edgecolor="white",
                            size=3, jitter=1, zorder=0, orient="h")
    sns.boxplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, color="black", width=.15, zorder=10,
                          showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                          showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                          saturation=1, orient="h")
    axes[0].set_title('Left condition', fontstyle='italic', fontweight = 'bold', fontsize = 20)
    axes[0].set_yticklabels(['MNE', 'Brainstorm'], fontsize = 15)
    axes[0].set_xlabel('', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('', fontweight='bold', fontsize=12)

    pt.half_violinplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, palette='pastel', bw=.2, cut=0.,
                                 scale="area", width=.6, inner=None, orient="h")
    sns.stripplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, palette='pastel', edgecolor="white",
                            size=3, jitter=1, zorder=0, orient="h")
    sns.boxplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, color="black", width=.15, zorder=10,
                          showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                          showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                          saturation=1, orient="h")
    axes[1].set_title('Right condition', fontstyle='italic', fontweight = 'bold', fontsize = 20)
    axes[1].set_yticklabels(['MNE', 'Brainstorm'], fontsize = 15)
    plt.xticks(fontsize = 15)
    axes[1].set_xlabel('Coherence values', fontsize=15)
    axes[1].set_ylabel('', fontweight='bold', fontsize=12)



'''
Function to plot all three together (MNE before, after and Brainstorm)
def rainplot_channel_f(dataMNE_L, dataMNE_R, dataMNE_L_f, dataMNE_R_f, dataBST_L, dataBST_R, name_channels, channel1, channel2, title):
    """
    Take connectivity values obtained from MNE and BST and for the 2 conditions for one subject
    as inputs, returns a rainplot of connectivity values for both conditions and toolboxes.
    The connectivity values considered are for 1 channel pair only, specified with its names, and each plot
    therefore comprises 144 points, one for each trial.
    ## inputs:
        # dataMNE_L: generally of the form "MNE_subject_1_L", connectivity values for a specific subject
                    and left condition obtained from MNE
        # dataMNE_R: generally of the form "MNE_subject_1_R", connectivity values for a specific subject
                    and right condition obtained from MNE
        # dataMNE_L_f: generally of the form "MNE_subject_1_L", connectivity values for a specific subject
                    and left condition obtained from MNE, for final version of MNE
        # dataMNE_R_f: generally of the form "MNE_subject_1_R", connectivity values for a specific subject
                    and right condition obtained from MNE, for final version of MNE
        # dataBST_L: connectivity values for a specific subject and left condition obtained from BST
        # dataBST_R: connectivity values for a specific subject and right condition obtained from BST
        # name_channels: list of channel labels
        # channel1: name of first channel composing channel pair interested in
        # channel2: name of second channel composing channel pair interested in
        # title : title that will be given to the graph
    ## output:
        # Four rainplots of coherence values across conditions (left and right) and toolboxes (BST and MNE) for a
        given subject and given channel pair
    """
    coorx = name_channels.index(channel1)
    coory = name_channels.index(channel2)
    data_R = []
    data_L = []
    for i in range(144):
        data_L.append(np.mean(dataMNE_L[i][coorx][coory]))  # Average across channels
        data_R.append(np.mean(dataMNE_R[i][coorx][coory]))
    for i in range(144):
        data_L.append(np.mean(dataBST_L[i][coorx][coory]))
        data_R.append(np.mean(dataBST_R[i][coorx][coory]))
    for i in range(144):
        data_L.append(np.mean(dataMNE_L_f[i][coorx][coory]))  # Average across channels
        data_R.append(np.mean(dataMNE_R_f[i][coorx][coory]))
    df = pd.DataFrame(data={'Toolbox': ['MNE'] * 144 + ['BST'] * 144 + ['MNEf'] * 144, 'Left condition': data_L, 'Right condition': data_R})


    f, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    pt.half_violinplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, palette='pastel', bw=.2, cut=0.,
                                 scale="area", width=.6, inner=None, orient="h")
    sns.stripplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, palette='pastel', edgecolor="white",
                            size=3, jitter=1, zorder=0, orient="h")
    sns.boxplot(ax = axes[0], x="Left condition", y="Toolbox", data=df, color="black", width=.15, zorder=10,
                          showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                          showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                          saturation=1, orient="h")
    axes[0].set_title('Left condition', fontstyle='italic', fontweight = 'bold', fontsize = 20)
    axes[0].set_yticklabels(["MNE \n ($\it{pre}$ $\it{parameters}$ \n $\it{modifications}$)", 'Brainstorm', "MNE \n ($\it{post}$ $\it{parameters}$ \n $\it{modifications}$)"], fontsize = 13)
    axes[0].set_xlabel('', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('', fontweight='bold', fontsize=12)

    pt.half_violinplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, palette='pastel', bw=.2, cut=0.,
                                 scale="area", width=.6, inner=None, orient="h")
    sns.stripplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, palette='pastel', edgecolor="white",
                            size=3, jitter=1, zorder=0, orient="h")
    sns.boxplot(ax = axes[1], x="Right condition", y="Toolbox", data=df, color="black", width=.15, zorder=10,
                          showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                          showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                          saturation=1, orient="h")
    axes[1].set_title('Right condition', fontstyle='italic', fontweight = 'bold', fontsize = 20)
    axes[1].set_yticklabels(["MNE \n ($\it{pre}$ $\it{parameters}$ \n $\it{modifications}$)", 'Brainstorm', "MNE \n ($\it{post}$ $\it{parameters}$ \n $\it{modifications}$)"], fontsize = 13)
    plt.xticks(fontsize = 15)
    axes[1].set_xlabel('Coherence values', fontsize=15)
    axes[1].set_ylabel('', fontweight='bold', fontsize=12)
    '''
