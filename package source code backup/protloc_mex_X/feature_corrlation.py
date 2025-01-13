
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from packaging import version

try:
    import scipy
    if version.parse(scipy.__version__) < version.parse('1.7.3'):
        warnings.warn("Your scipy version is older than 1.7.3 and may not operate correctly.")
    from scipy import stats
except ImportError:
    warnings.warn("Scipy is not installed. Some features may not work as expected.")


class spear_correlation():
    def __init__(self,X_data,y_data):
        self.X_data=X_data
        self.y_data=y_data
    @staticmethod
    def plot_spearman_heatmap(X_data,save_path_completed,draw_heatmap=True,figure_size=(10,10)):
        corrs, p_values = stats.spearmanr(X_data)
        mask = np.zeros_like(corrs)
        mask[np.triu_indices_from(mask)] = True
        if draw_heatmap:
            fig, ax = plt.subplots(figsize=figure_size)
            # Set image size
            fig=sns.heatmap(
                corrs, vmin=-1, vmax=1, center=0, mask=mask, square=True,
                cmap='viridis',
                xticklabels=X_data.columns, 
                yticklabels=X_data.columns)
            fig.set_title('Spearman Correlation Heatmap')
            fig.figure.savefig(save_path_completed,
                       dpi=1000, bbox_inches="tight")
            plt.close(fig.figure)
        # Create DataFrame objects to store correlation coefficients and P-values
        df_corr = pd.DataFrame(corrs, index=X_data.columns, columns=X_data.columns)
        df_p = pd.DataFrame(p_values, index=X_data.columns, columns=X_data.columns)
        return df_corr,df_p
    
    def feature_crossover_reg(self, feature_cross_over, save_path, figure_size=(13,6)):
        for i_name in feature_cross_over:
            # calculate spearman correlation and p-value
            corr, p_value = stats.spearmanr(self.X_data[i_name.split('*_')[0]], self.X_data[i_name.split('*_')[1]])
            # create the plot
            fig, axs = plt.subplots(figsize=figure_size)
            fig = sns.regplot(x=self.X_data[i_name.split('*_')[0]], y=self.X_data[i_name.split('*_')[1]], 
                ax=axs, scatter_kws={'alpha':0.3},
                line_kws={'color':'g'})
            axs.set_xlabel(i_name.split('*_')[0], fontsize=13)
            axs.set_ylabel(i_name.split('*_')[1], fontsize=13)
            fig.set_title('Feature Crossover Regression Analysis\nspearman: {:.3f}, p-value: {:.4f}'.format(corr, p_value))
            fig.figure.savefig(save_path+i_name.split('*_')[0]+'vs_'+i_name.split('*_')[1]+' crossover_reg.pdf',
                               dpi=1000, bbox_inches="tight")
            plt.close(fig.figure)
