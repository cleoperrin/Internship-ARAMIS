import numpy as np
from mne.stats import bonferroni_correction, fdr_correction

def pval_correction(pvalues, correction_method):
    """
     Take p-values of statistical test as inputs and returns corrected p-values through wanted correction,
     bonferroni or fdr.
     ## inputs:
         # pvalues: generally of the form "t_test_toolboxes_subj1_L.pvalue" or "results_wil_test_L",
                    pvalues to be corrected
         # correction_method: correction method to be used, either bonferroni or fdr
     ## output:
         # Corrected p-values based on wanted correction (The self-to-self diagonal was removed, so be careful as
            only the lower triangle of the matrix is still well organized based on channels !!)
     """
    # Removing NaN diagonal to be able for matrix to be treated and correction method to return corrected p-values
    NaN_removed = pvalues[~np.eye(pvalues.shape[0], dtype=bool)].reshape(pvalues.shape[0], -1)
    # Computing correction with MNE functions
    if correction_method == 'bonferroni':
        corrected_pval = bonferroni_correction(NaN_removed, alpha=0.05)[1]
    elif correction_method == 'fdr':
        corrected_pval = fdr_correction(NaN_removed, alpha=0.05, method='indep')[1]

    return corrected_pval


















'''
#plot direct, mais pas forcément utile
def compute_plot_correction(pvalues, correction_method, name_channels, title):
    # Removing NaN diagonal to be able for matrix to be treated and correction method to return corrected p-values
    NaN_removed = pvalues[~np.eye(pvalues.shape[0], dtype=bool)].reshape(pvalues.shape[0],
                                                                         -1)  # removing the NaN diagonal
    # Computing correction with MNE functions
    if correction_method == 'bonferroni':
        corrected_pval = bonferroni_correction(NaN_removed, alpha=0.05)[1]
    elif correction_method == 'fdr':
        corrected_pval = fdr_correction(NaN_removed, alpha=0.05, method='indep')[1]

    # Plot the correction
    mask = np.triu(np.ones_like(corrected_pval, dtype=bool))  # Generate a mask for the upper triangle
    map = sns.heatmap(corrected_pval, annot=False, mask=mask, cbar_kws={'label': 'p-value'}, square=True, vmin=0,
                      vmax=0.05, cmap="GnBu_r", linewidths=.5)
    map.set_title(title, fontweight='bold')
    map.set_xticklabels(name_channels)
    map.set_yticklabels(name_channels)
    plt.show()
#Conduct the correction, specifying which type : bonferroni or fdr, and on which p-values
compute_plot_correction(t_test_toolboxes_subj1_L.pvalue, 'bonferroni', channel_names, 'Corrected p-value (bonferroni) of two-sided t-test between connectivity \n results provided by mscoh MNE and mscoh BST (subject 1, left)')
'''