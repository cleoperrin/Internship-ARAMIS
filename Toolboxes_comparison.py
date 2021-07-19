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

# CHANNEL NAMES
channels = loadmat('/Users/cleo.perrin/Desktop/Brainstorm_MatLab/RIGOLETTO_bst_project/data/IROS2020/data_A01/channel.mat', squeeze_me=True)
channel_names = []
for i in range(22):
    channel_names.append(channels['Channel'][i][0])
#OR : channel_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

# CONNECTIVITY RESULTS BST
data = loadmat('/Users/cleo.perrin/Documents/Database_MOABB/BNCI2014001/Coh_2222288.mat', squeeze_me=True)
Matrices = data['MatCoh']

for k in range(len(Matrices)):
    exec(f'BST_subject_{k + 1} = Matrices[k][0]')

    # Reorganize matrix 22x22x288 to 288x22x22 (MatLab to Python transfer)
    exec(f'BST_subject_{k + 1} = np.ascontiguousarray(BST_subject_{k + 1}.T)')

    # Create empty arrays to be filled depending on the condition (l or r)
    exec(f'BST_subject_{k + 1}_R = np.empty((int(len(Matrices[k][1])/2),len(channel_names),len(channel_names)))')
    exec(f'BST_subject_{k + 1}_L = np.empty((int(len(Matrices[k][1])/2),len(channel_names),len(channel_names)))')

    # Fill r array with connectivity data of r condition, same with l condition
    exec(f"""
i = 0
j = 0
for n_trial in range(len(Matrices[k][1])):
    if Matrices[k][1][n_trial] == 'right_hand                      ' :
        BST_subject_{k + 1}_R[i] = BST_subject_{k + 1}[n_trial] 
        i +=1
    elif Matrices[k][1][n_trial] == 'left_hand                       ' :
        BST_subject_{k + 1}_L[j] = BST_subject_{k + 1}[n_trial]
        j += 1
        """)

    # Delete variables comprising data for both conditions as not needed
    exec(f'del BST_subject_{k + 1}')

# CONNECTIVITY RESULTS MNE
# To comment / uncomment based on the FC results wanted
for k in range(9):
    # For MNE coh multitaper
#    exec(f'MNE_subject_{k + 1}_L = ((np.load("Data/MNE_FC_SingleTrial_001-2014_Subj{k+1}_L.npy", allow_pickle = True)).item())["coh"]')
#    exec(f'MNE_subject_{k + 1}_R = ((np.load("Data/MNE_FC_SingleTrial_001-2014_Subj{k + 1}_R.npy", allow_pickle = True)).item())["coh"]')
    # For MNE mscoh multitaper
#    exec(f'MNE_subject_{k + 1}_L = ((np.load("DataMSCOH/MNE_FC_SingleTrial_001-2014_Subj{k+1}_L.npy", allow_pickle = True)).item())["mscoh"]')
#    exec(f'MNE_subject_{k + 1}_R = ((np.load("DataMSCOH/MNE_FC_SingleTrial_001-2014_Subj{k + 1}_R.npy", allow_pickle = True)).item())["mscoh"]')
    # For MNE mscoh fourier
#    exec(f'MNE_subject_{k + 1}_L = ((np.load("DataMSCOH_FOURIER/MNE_FC_SingleTrial_001-2014_Subj{k+1}_L.npy", allow_pickle = True)).item())["mscoh"]')
#    exec(f'MNE_subject_{k + 1}_R = ((np.load("DataMSCOH_FOURIER/MNE_FC_SingleTrial_001-2014_Subj{k + 1}_R.npy", allow_pickle = True)).item())["mscoh"]')
    # For MNE mscoh fourier with changed time window
#    exec(f'MNE_subject_{k + 1}_L = ((np.load("DataMSCOH_FOURIER_timewin/MNE_FC_SingleTrial_001-2014_Subj{k + 1}_L.npy", allow_pickle = True)).item())["mscoh"]')
#    exec(f'MNE_subject_{k + 1}_R = ((np.load("DataMSCOH_FOURIER_timewin/MNE_FC_SingleTrial_001-2014_Subj{k + 1}_R.npy", allow_pickle = True)).item())["mscoh"]')
    # For MNE mscoh fourier with changed time window and frequencies
    exec(f'MNE_subject_{k + 1}_L = ((np.load("DataMSCOH_FOURIER_timewin_freq/MNE_FC_SingleTrial_001-2014_Subj{k + 1}_L.npy", allow_pickle = True)).item())["mscoh"]')
    exec(f'MNE_subject_{k + 1}_R = ((np.load("DataMSCOH_FOURIER_timewin_freq/MNE_FC_SingleTrial_001-2014_Subj{k + 1}_R.npy", allow_pickle = True)).item())["mscoh"]')


# TTEST BETWEEN CONDITIONS FOR EACH SUBJECT (matrices 9 x 22 x 22) AND THE 2 TOOLBOXES
ttest_cond_MNE = np.empty((9, 22, 22))
ttest_cond_BST = np.empty((9, 22, 22))
for k in range(9):
    exec(f'ttest_cond_MNE[k] = stats.ttest_rel(MNE_subject_{k + 1}_L,MNE_subject_{k + 1}_R).pvalue')
    exec(f'ttest_cond_BST[k] = stats.ttest_rel(BST_subject_{k + 1}_L,BST_subject_{k + 1}_R).pvalue')

# TTEST BETWEEN THE 2 TOOLBOXES FOR EACH SUBJECT AND THE 2 CONDITIONS
ttest_tbxs_L = np.empty((9, 22, 22))
ttest_tbxs_R = np.empty((9, 22, 22))
for k in range(9):
    exec(f'ttest_tbxs_L[k] = stats.ttest_rel(MNE_subject_{k + 1}_L,BST_subject_{k + 1}_L).pvalue')
    exec(f'ttest_tbxs_R[k] = stats.ttest_rel(MNE_subject_{k + 1}_R,BST_subject_{k + 1}_R).pvalue')

# CREATION OF MATRICES COMPRISING MEAN OF COHERENCE VALUES ACROSS TRIALS FOR BOTH CONDITION AND TOOLBOXES FOR EACH SUBJECT
mean_all_L_MNE, mean_all_R_MNE = np.empty((9, 22, 22)), np.empty((9, 22, 22))
mean_all_L_BST, mean_all_R_BST = np.empty((9, 22, 22)), np.empty((9, 22, 22))
for k in range(9):
    exec(f'mean_all_L_BST[k] = np.mean(BST_subject_{k + 1}_L, axis = 0 )')
    exec(f'mean_all_R_BST[k] = np.mean(BST_subject_{k + 1}_R, axis = 0 )')
    exec(f'mean_all_L_MNE[k] = np.mean(MNE_subject_{k + 1}_L, axis = 0 )')
    exec(f'mean_all_R_MNE[k] = np.mean(MNE_subject_{k + 1}_R, axis = 0 )')

# WILCOXON BETWEEN CONDITIONS FOR EACH TOOLBOXES and BETWEEN TOOLBOXES FOR EACH CONDITIONS (N = 9 SUBJECTS)
# Can not do t-test as only sample of size 9, and Wilcoxon only works on 1D array, so will
# add one by one p-values of Wil test
wlcx_cond_MNE, wlcx_cond_BST = np.empty((22, 22)), np.empty((22, 22)) # For stat test btwn conditions
wlcx_tbxs_L, wlcx_tbxs_R = np.empty((22, 22)), np.empty((22, 22)) # For stat test btwn toolboxes
for i in range(22):
    for j in range(22):
        # Create empty lists (1D) that for one channel pair will comprise connectivity
        # values of that channel pair for each 9 subjects
        value_ij_L_MNE = []
        value_ij_R_MNE = []
        value_ij_L_BST = []
        value_ij_R_BST = []
        for k in range(9): # Looping through subjects
            value_ij_L_MNE.append(mean_all_L_MNE[k][i][j])
            value_ij_R_MNE.append(mean_all_R_MNE[k][i][j])
            value_ij_L_BST.append(mean_all_L_BST[k][i][j])
            value_ij_R_BST.append(mean_all_R_BST[k][i][j])
        if value_ij_L_BST == value_ij_L_MNE: # Wilcoxon unable to treat the pb if all values are identical.
            wlcx_tbxs_L[i][j] = np.nan
            wlcx_tbxs_R[i][j] = np.nan
            wlcx_cond_MNE[i][j] = np.nan
            wlcx_cond_BST[i][j] = np.nan
        else : #If can conduct the test, add p-value of the test in the array
            wlcx_cond_MNE[i][j] = (stats.wilcoxon(value_ij_L_MNE, value_ij_R_MNE)).pvalue
            wlcx_cond_BST[i][j] = (stats.wilcoxon(value_ij_L_BST, value_ij_R_BST)).pvalue
            wlcx_tbxs_L[i][j] = (stats.wilcoxon(value_ij_L_MNE, value_ij_L_BST)).pvalue
            wlcx_tbxs_R[i][j] = (stats.wilcoxon(value_ij_R_MNE, value_ij_R_BST)).pvalue
        del value_ij_L_MNE, value_ij_L_BST, value_ij_R_MNE, value_ij_R_BST


# CONDUCTING THE STEPS OF THE COMPARISON

# FOR A FIRST APPROACH AND VISUALISATION

# Function plotting p-values of t-test between conditions for MNE (left) and BST (right)
from Pipeline import plot_ttest_con_tbx
plot_ttest_con_tbx(ttest_cond_MNE[0],ttest_cond_BST[0],channel_names,'P-value of two-sided t-test between left and right \n  conditions of all-to-all coherence in sensor space \n (subject 1)')

# Plotting p-values of wilcoxon test between conditions for each toolboxes, and return channel pairs for which
# the difference of connectivity between left and right conditions is the most significant. These results can be
# used in rainplot_channel() to focus on the most important channels with regards to the research question
from Pipeline import plot_wil_con_tbx
plot_wil_con_tbx(wlcx_cond_MNE, wlcx_cond_BST, channel_names, 'P-value of Wilcoxon signed-rank test \n between left and right conditions. \n (n = 9 subjects)')


# FOR A GROUP COMPARISON

# Function plotting p-values of wilcoxon test between toolboxes for both conditions
from Pipeline import plot_group_wil_tbx
plot_group_wil_tbx(wlcx_tbxs_L, wlcx_tbxs_R, channel_names, 'P-value of Wilcoxon test between MNE and BST toolboxes. \n (n = 9 subjects)')


# BASED ON THESE FIRST RESULTS, CONDUCT FURTHER ANALYSIS AT THE INDIVIDUAL SCALE

# Function that will compute and plot the difference of coherence values across sensors for a given subject
from Pipeline import diff_conn
diff_conn(MNE_subject_1_L, MNE_subject_1_R, BST_subject_1_L, BST_subject_1_R, channel_names,'Difference (MNE - BST) of mean coherence values (in all-to-all sensors space) \n provided by mscoh MNE and BST \n (subject 1)')  # be careful to give L and L or R and R but not L and R

# Function computing and plotting p-values of t-test computed between the 2 toolboxes for 1 subject and 1 condition.
from Pipeline import pvalues_ttest_tbx
pvalues_ttest_tbx(MNE_subject_1_L, MNE_subject_1_R, BST_subject_1_L, BST_subject_1_R, channel_names, 'P-values of two-sided t-test between \n coherence results given by MNE vs BST. \n Subject 1, left condition')

# Function plotting as boxplots coherence values for 1 subject, for each conditions and toolboxes
from Pipeline import rainplot_tbx_cond
rainplot_tbx_cond(MNE_subject_1_L, MNE_subject_1_R, BST_subject_1_L, BST_subject_1_R, 'Rainplot of coherence values across \n toolboxes (mscoh MNE and BST) and conditions for subject 1')

# Function plotting connectivity values for 1 subject for both conditions and toolboxes for 1 channel pair
# Interesting to test on channel pairs given by plot_wil_con_tbx()
from Pipeline import rainplot_channel
rainplot_channel(MNE_subject_1_L, MNE_subject_1_R, BST_subject_1_L, BST_subject_1_R, channel_names, 'CP4', 'FC3', 'Rainplot of coherence values across toolboxes and \n conditions for subject 1, channel pair CP4 / FC3.')


# P-VAL CORRECTION
# Conduct the correction, specifying which type : 'bonferroni' or 'fdr', and on which p-values.
# Corrected p-values array can be passed on to be plotted in the functions
from Correction import pval_correction
pval_correction(ttest_cond_MNE[0], 'bonferroni')


