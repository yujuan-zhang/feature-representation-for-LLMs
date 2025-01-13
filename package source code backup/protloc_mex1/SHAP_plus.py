# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:23:53 2023

@author: qq102
"""
import pandas as pd
import numpy as np
import re
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from packaging import version
import warnings

try:
    import sklearn
    sklearn_exists = True
except ImportError:
    sklearn_exists = False
    
if sklearn_exists:
    if version.parse(sklearn.__version__) < version.parse('1.0.2'):
        warnings.warn("Your sklearn version is older than 1.0.2  +\
                      and may be not operation correct.")
    from sklearn.preprocessing import MinMaxScaler
else:
    warnings.warn("Sklearn not found. MinMaxScaler will not be available.") 

# from sklearn.preprocessing import MinMaxScaler




##库核心框架
class SHAP_importance_sum:
    def __init__(self):
        self.shapley_data_all=None
    
    @staticmethod
    def str_contain(str_input,input_names):
        pattern=re.compile(str_input)
        str_outcom=[]
        for i in input_names:
            if re.search(pattern,i):
                str_outcom.append(i)
        return(str_outcom)
    
    @staticmethod
    def feature_shap_statistic(SHAP_data_all,depleted_ID_len=0):
        SHAP_data_all_feature_shap=list(map(lambda x:SHAP_data_all[x].iloc[:,:len(SHAP_data_all[x].columns)-depleted_ID_len],SHAP_data_all.keys()))
        SHAP_data_all_feature_shap=dict(zip(list(SHAP_data_all.keys()),SHAP_data_all_feature_shap))
        for i,value in enumerate(SHAP_data_all_feature_shap):
            feature_inner_name=SHAP_data_all_feature_shap[value].columns
            SHAP_data_all_feature_shap[value]=list(map(lambda x:pd.DataFrame(SHAP_data_all_feature_shap[value][x]),SHAP_data_all_feature_shap[value].columns))
            SHAP_data_all_feature_shap[value]=dict(zip(list(feature_inner_name),SHAP_data_all_feature_shap[value]))
            for i_1,value_1 in enumerate(SHAP_data_all_feature_shap[value]):
                SHAP_data_all_feature_shap[value][value_1].columns=[value]
        SHAP_data_all_feature_shap_outcome=dict()
        
        for i in list(feature_inner_name):
            SHAP_data_all_feature_shap_outcome[i]=list(map(lambda x: SHAP_data_all_feature_shap[x][i],SHAP_data_all_feature_shap.keys()))   
              
            SHAP_data_all_feature_shap_outcome[i]=reduce(lambda x,y: pd.concat([x,y],axis=1),SHAP_data_all_feature_shap_outcome[i])       
            
        return(SHAP_data_all_feature_shap_outcome)
    
    def IG_importance_sum_claulate_process(self,depleted_ID_len=0,file_name='human_'):
        """
        Calculates and processes Feature_aggregation_value(SHAP/EBM/IG)values to assess the overall importance of features. 
        This function first sums the Feature_aggregation_value values,
        then normalizes the summation results for potential different type of Feature_aggregation_value comparison(not feature importance or prediction probability). 
        It also computes the mean absolute value of Feature_aggregation_value for each feature to represent
        their importance(the same as the SHAP package), and compiles this data into a final DataFrame.

        Parameters:
            depleted_ID_len (int): The number of columns to exclude from the feature data.
            file_name (str): The file prefix used for saving file.

        Returns:
            dict: A dictionary containing the normalized SHAP summation results, feature importance, and its transposed version.

        Note:
            This version's scale returns scale(normalized) value not probabilities
        """

        scale=MinMaxScaler()
        self.type_data_shap_sum_all=dict()
        self.type_data_shap_sum_all_outcome=dict()
        self.shap_feature_importance=dict()
        self.shap_feature_importance_T=None
        for value in self.shapley_data_all.keys():
            ##shap_value summation
            
            self.type_data_shap_sum_all[value]=np.array(self.shapley_data_all[value].iloc[:,:len(self.shapley_data_all[value].columns)-depleted_ID_len]).sum(axis=1)
            self.type_data_shap_sum_all[value]=pd.DataFrame({'sum_shap':self.type_data_shap_sum_all[value].reshape(-1,),
                                                        'scale':scale.fit_transform(self.type_data_shap_sum_all[value].reshape(-1,1)).reshape(-1,)},
                                                        index=self.shapley_data_all[value].index)
            self.type_data_shap_sum_all_outcome[value]=pd.merge(self.shapley_data_all[value],self.type_data_shap_sum_all[value],how="inner",left_index=True,right_index=True)
            
            
            ##sum mean absolute feature value as feature importance
            self.shap_feature_importance[value]=np.abs(self.shapley_data_all[value].iloc[:,:len(self.shapley_data_all[value].columns)-depleted_ID_len])
            self.shap_feature_importance[value]=self.shap_feature_importance[value].mean(axis=0)
            self.shap_feature_importance[value]=pd.DataFrame(self.shap_feature_importance[value],columns=[value])

        self.shap_feature_importance=reduce(lambda x,y: pd.merge(x,y,how="inner",left_index=True,right_index=True),self.shap_feature_importance.values())

        self.shap_feature_importance.columns=[re.compile('('+file_name+')|(_shap_value)').sub('',name) for name in self.shap_feature_importance.columns]
        ##shap_feature_importance transposed
        self.shap_feature_importance_T=self.shap_feature_importance.T
        return{'type_data_shap_sum_all_outcome':self.type_data_shap_sum_all_outcome,
                'shap_feature_importance':self.shap_feature_importance,
                'shap_feature_importance_T':self.shap_feature_importance_T}
    
    
    def SHAP_importance_sum_claulate_process(self, base_probablity_dict, depleted_ID_len=0, file_name='human_'):
        """
        Calculates and processes SHAP value(only support SHAP !!!)values to assess the overall importance of features, while considering baseline probabilities
        (base_probablity). This function first sums the SHAP values, then adds the baseline probability to each sample's SHAP summation,
        and normalizes the result (this result can be see as the prediction probability). 
        It also computes the mean absolute value of SHAP values for each feature to represent 
        their importance(the same as the SHAP package),
        and compiles this data into a final DataFrame.

        Parameters:
            base_probablity_dict (dict): A dictionary storing the baseline probability for each category.
            depleted_ID_len (int): The number of columns to exclude from the feature data.
            file_name (str): The file prefix used for regex processing of column names.

        Returns:
            dict: A dictionary containing the normalized SHAP summation results, feature importance, and its transposed version.

        Note:
            This version's scale returns probabilities that are based on base_value, i.e., it considers the impact of the
            baseline probability on the calculation of probabilities.
        """

        scale = MinMaxScaler()
        self.type_data_shap_sum_all = dict()
        self.type_data_shap_sum_all_outcome = dict()
        self.shap_feature_importance = dict()
        self.shap_feature_importance_T = None
        for value in self.shapley_data_all.keys():
            # shap_value summation
            sum_shap = np.array(self.shapley_data_all[value].iloc[:, :len(self.shapley_data_all[value].columns)-depleted_ID_len]).sum(axis=1)

            # get corresponding base_probablity
            base_value = base_probablity_dict[value]

            # create sum_shap copy vision for calculating probablity value
            probablity = sum_shap.copy()

            # add base_probablity to probablity value
            probablity += base_value

            self.type_data_shap_sum_all[value] = pd.DataFrame({'sum_shap': sum_shap.reshape(-1,),
                                                               'scale': scale.fit_transform(sum_shap.reshape(-1,1)).reshape(-1,),
                                                               'probablity': probablity},  # 新增probablity列
                                                              index=self.shapley_data_all[value].index)
            self.type_data_shap_sum_all_outcome[value] = pd.merge(self.shapley_data_all[value], self.type_data_shap_sum_all[value], how="inner", left_index=True, right_index=True)

            # sum mean absolute feature value as feature importance
            self.shap_feature_importance[value] = np.abs(self.shapley_data_all[value].iloc[:, :len(self.shapley_data_all[value].columns)-depleted_ID_len])
            self.shap_feature_importance[value] = self.shap_feature_importance[value].mean(axis=0)
            self.shap_feature_importance[value] = pd.DataFrame(self.shap_feature_importance[value], columns=[value])

        self.shap_feature_importance = reduce(lambda x, y: pd.merge(x, y, how="inner", left_index=True, right_index=True), self.shap_feature_importance.values())

        self.shap_feature_importance.columns = [re.compile('('+file_name+')|(_shap_value)').sub('', name) for name in self.shap_feature_importance.columns]
        ##shap_feature_importance transposed
        self.shap_feature_importance_T = self.shap_feature_importance.T
        return {'type_data_shap_sum_all_outcome': self.type_data_shap_sum_all_outcome,
                'shap_feature_importance': self.shap_feature_importance,
                'shap_feature_importance_T': self.shap_feature_importance_T}
    
    
    def calculate_feature_shap_bins(self, keys, feature, bins_num=10, q_num=10):
        result = pd.DataFrame()
        for key in keys:
            data = self.shapley_data_all[key]
            feature_data = data[feature]
            # 先按分位数分箱
            quantiles = pd.qcut(feature_data, q=q_num, duplicates='drop')
            # 记录全局箱子编号
            global_bins = 0
            bins_list = []
            # 在每个分位数区间内按等距分箱
            for _, group in feature_data.groupby(quantiles):
                # 由于group的index是feature_data的index，所以可以通过这个index从feature_data中取到对应的值
                group_values = feature_data[group.index]
                bins = pd.cut(group_values, bins=bins_num, include_lowest=True)
                # 获取箱子编号，并加上全局箱子编号
                bins_codes = bins.cat.codes + global_bins
                bins_list.append(bins_codes)
                # 更新全局箱子编号
                global_bins = bins_codes.max() + 1
            # 拼接所有的箱子编号
            bins_codes = pd.concat(bins_list)
            temp_df = pd.DataFrame({
                f'shap_values_{key}': feature_data,
                f'bins_{key}': bins_codes
            })
            if result.empty:
                result = temp_df
            else:
                result = result.join(temp_df, how='outer')
        return result
    


    

class FeaturePlotSource:
    def __init__(self, X_data, shapley_data_all):
        self.X_data = X_data
        self.shapley_data_all = shapley_data_all
    
    def plot(self):
        pass


class FeaturePlot(FeaturePlotSource):

    class ShapImportancePlot:
        def __init__(self, func):
            self.func = func
            self.self_obj = None
        
        def __get__(self, instance, owner):
            self.self_obj = instance
            return self
        
        def __call__(self, *args, **kwargs):
            self.plot(*args, **kwargs)
        
        def plot(self, file_name, save_path, plot_size, *args, **kwargs):
            shap_values_plot = [value for value in self.self_obj.shapley_data_all.values()]
            shap_values_plot = [value.to_numpy() if not isinstance(value, np.ndarray) 
                                else value for value in shap_values_plot]
            shap_values_class = [name for name in self.self_obj.shapley_data_all.keys()]
            
            plot_shap_importance_outcom = plt.figure(figsize=plot_size, dpi=1000)
            plt.title(file_name + '_shap_importance')
            shap.summary_plot(shap_values_plot, self.self_obj.X_data, plot_type="bar", class_names=shap_values_class, 
                              show=False, plot_size=plot_size, *args, **kwargs)
            plot_shap_importance_outcom.savefig(save_path + file_name + '_shap_importance.png', dpi=1000, bbox_inches="tight")
            plot_shap_importance_outcom.savefig(save_path + file_name + '_shap_importance.pdf', dpi=1000, bbox_inches="tight")
            plt.close()

    @ShapImportancePlot
    def shap_importance_plot(self, file_name, save_path, plot_size, *args, **kwargs):
        pass


    class FeatureSummaryPlot:
        def __init__(self, func):
            self.func = func
            self.self_obj = None
        
        def __get__(self, instance, owner):
            self.self_obj = instance
            return self
        def __call__(self, *args, **kwargs):
            self.plot(*args, **kwargs)
        def shap_summary_plot(self, shapley_data_all, X_input, class_names, *args, **kwargs):
            plot_shap_summary_outcom = plt.figure(figsize=self.plot_size, dpi=1000)
            plt.title(self.file_name + "_" + class_names + '_shap_summary')
            shap.summary_plot(shapley_data_all, X_input, plot_type="dot", show=False, plot_size=self.plot_size, *args, **kwargs)
            plot_shap_summary_outcom.savefig(self.save_path + class_names + self.file_name + "_shap_summary.png", dpi=1000, bbox_inches="tight")
            plot_shap_summary_outcom.savefig(self.save_path + class_names + self.file_name + "_shap_summary.pdf", dpi=1000, bbox_inches="tight")
            plt.close()

        def plot(self, file_name, save_path,plot_size,*args, **kwargs):
            self.file_name = file_name
            self.save_path = save_path
            self.plot_size = plot_size
            shap_values_plot = [value for value in self.self_obj.shapley_data_all.values()]
            shap_values_plot = [value.to_numpy() if not isinstance(value, np.ndarray) 
                                else value for value in shap_values_plot]
            shap_values_class = [name for name in self.self_obj.shapley_data_all.keys()]
            for i, value in enumerate(shap_values_class):
                self.shap_summary_plot(shapley_data_all=shap_values_plot[i], X_input=self.self_obj.X_data,
                                        class_names=value, *args, **kwargs)
    
    @FeatureSummaryPlot
    def Shapley_summary_plot(self, file_name, save_path, plot_size, *args, **kwargs):
        pass
    
    class FeatureHistPlot:
        def __init__(self, func):
            self.func = func
            self.self_obj = None
        
        def __get__(self, instance, owner):
            self.self_obj = instance
            return self
        def __call__(self, *args, **kwargs):
            self.plot(*args, **kwargs)
        def plot(self,shap_indicate_feature,save_path, file_name, png_plot=True,
                 jointplot_kwargs=None, marg_x_kwargs=None, marg_y_kwargs=None):
            self.shap_indicate_feature = shap_indicate_feature
            self.save_path = save_path
            self.file_name = file_name
            self.png_plot = png_plot
            
            if jointplot_kwargs is None:
                jointplot_kwargs = {}
            if marg_x_kwargs is None:
                marg_x_kwargs = {}
            if marg_y_kwargs is None:
                marg_y_kwargs = {}
                
            for value in list(self.self_obj.shapley_data_all.keys()):
                for i_name in self.shap_indicate_feature: 
                    df = pd.DataFrame({'Feature_value':self.self_obj.X_data[i_name],
                                       'shapley_value':self.self_obj.shapley_data_all[value][i_name]})
                    ax = sns.jointplot(x=df['Feature_value'], y=df['shapley_value'],
                                       kind='hist', color='r',**jointplot_kwargs)
                    
                    ax.set_axis_labels(xlabel=f"Feature value of {i_name}",
                                       ylabel='shapley_value')
                    # # 设置直方图bin的数量
                    # ax.ax_marg_x.hist(df['Feature_value'], **marg_x_kwargs)
                    # ax.ax_marg_y.hist(df['shapley_value'], **marg_y_kwargs,orientation='horizontal')
                    
                    # 使用 Matplotlib 绘制坐标轴上的直方图
                    fig = plt.gcf()
                    ax1 = fig.add_axes(ax.ax_marg_x.get_position())
                    ax1.hist(df['Feature_value'], **marg_x_kwargs)
                    ax1.set_xticks([])
                    ax2 = fig.add_axes(ax.ax_marg_y.get_position())
                    ax2.hist(df['shapley_value'], orientation='horizontal',**marg_y_kwargs)
                    ax2.set_yticks([])
                    fig.suptitle(f"{value} {self.file_name}",y=1.05)
                    ax.savefig(self.save_path + 
                               f"{self.file_name}_{value}_{i_name}_histplot.pdf",
                               dpi=1000, bbox_inches="tight")
                    if self.png_plot:
                        ax.savefig(self.save_path + 
                                   f"{self.file_name}_{value}_{i_name}_histplot.png",
                                   dpi=1000, bbox_inches="tight")
                    plt.close(ax.figure)
    @FeatureHistPlot
    def feature_hist_plot(self,shap_indicate_feature,save_path,
                                file_name, png_plot,*args, **kwargs):
        pass
    
    
    
    
    class FeatureScatterplot(FeaturePlotSource):
        def __init__(self, func):
            self.func = func
            self.self_obj = None
        
        def __get__(self, instance, owner):
            self.self_obj = instance
            return self
        def __call__(self, *args, **kwargs):
            self.plot(*args, **kwargs)
            
        def plot(self,shap_indicate_feature,save_path,file_name='human_train',png_plot=True,dependence=True,*args, **kwargs):
            self.shap_indicate_feature = shap_indicate_feature
            self.save_path = save_path
            self.file_name = file_name
            self.png_plot = png_plot
            for value in list(self.self_obj.shapley_data_all.keys()):
                for i_name in self.shap_indicate_feature:
                    plt.figure()
                    if dependence:
                        shap.dependence_plot(i_name.split('*_')[0], np.array(self.self_obj.shapley_data_all[value]), 
                                             self.self_obj.X_data, title=self.file_name+"_"+value+"_"+'shap_dependence', 
                                             show=False, interaction_index=i_name.split('*_')[1],*args, **kwargs)
                        plt.savefig(self.save_path+self.file_name+"_"+value+"_"+i_name.split('*_')[0]+'vs_'+i_name.split('*_')[1]+"_shap_dependence.pdf",dpi=1000,bbox_inches="tight")
                        if self.png_plot:
                            plt.savefig(self.save_path+self.file_name+"_"+value+"_"+i_name.split('*_')[0]+'vs_'+i_name.split('*_')[1]+"_shap_dependence.png",dpi=1000,bbox_inches="tight")
                    else:
                        shap.dependence_plot(i_name, np.array(self.self_obj.shapley_data_all[value]), self.self_obj.X_data,
                                             title=self.file_name+"_"+value+"_"+'shap_independence', show=False,
                                             interaction_index=None,*args, **kwargs)
                        plt.savefig(self.save_path+self.file_name+"_"+value+"_"+i_name+"_shap_nodependence.pdf",dpi=1000,bbox_inches="tight")
                        if self.png_plot:
                            plt.savefig(self.save_path+self.file_name+"_"+value+"_"+i_name+"_shap_nodependence.png",dpi=1000,bbox_inches="tight")
                    plt.close()

    @FeatureScatterplot
    def feature_scatter_plot(self,shap_indicate_feature,save_path,
                                file_name, png_plot,dependence,*args, **kwargs):
        pass        
        

   

class LocalAnalysisPlot(FeaturePlotSource):
    def __init__(self, X_data, shapley_data_all,probability_data):
        super().__init__(X_data, shapley_data_all)
        self.probability_data = probability_data

     
    def plot(self,protein_indicate,save_path, file_name='huamn', plot_size=(10,10),png_plot=True,
             feature_remain=5,positive_color='#FFB6C1',negative_color='#ADD8E6'):
        # protein_indicate= protein_indicate
        # save_path = save_path
        # file_name = file_name
        # png_plot = png_plot
        
            
        for value in list(self.shapley_data_all.keys()):
            for i_name in protein_indicate:
                ##创建绘图概率
                probability_predict=self.probability_data[value].loc[i_name].to_numpy()[0]
                
                ##创建绘图列表
                df = pd.DataFrame({'Feature_value':self.X_data.loc[i_name],
                                   'shapley_value':self.shapley_data_all[value].loc[i_name]})
                df['shapley_value_abs'] = df['shapley_value'].abs()
                df= df.sort_values(by='shapley_value_abs', ascending=False).iloc[:feature_remain]
                df['abs_rank'] = df['shapley_value_abs'].rank( ascending=True)
                feature_name = df.index.tolist()
                feature_value=df['Feature_value'].tolist()
                yticklabels = [f"{feature_name[i]} ({feature_value[i]:.4f})" for i in range(len(feature_name))]
                # 将正数和负数数据分别存储到两个不同的数组中
                # 使用条件语句将正数和负数数据分别存储到两个不同的DataFrame中
                positive_data = df[df['shapley_value'] > 0]
                negative_data = df[df['shapley_value'] <= 0]
                
                # 创建一个包含两个水平直方图的子图
                fig, ax = plt.subplots()

                # 绘制正数数据的水平直方图
                ax.barh(positive_data['abs_rank'].to_numpy(), 
                        positive_data['shapley_value'], align='center', color=positive_color)

                # 绘制负数数据的水平直方图
                ax.barh(negative_data['abs_rank'].to_numpy(), 
                        negative_data['shapley_value'], align='center', color=negative_color)
                ax.set_yticks(df['abs_rank'].to_numpy())
                ax.set_yticklabels(yticklabels)
                
                # 添加概率预测值的注释
                ax.text(1, 1, f"{i_name} for {value} Probability Prediction: {probability_predict:.2f}", 
                transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right')
                ax.set_xlabel('shapley_value')
                ax.set_ylabel(f"Feature value of {i_name}")

                fig.set_size_inches(plot_size)
                fig.savefig(save_path + 
                           f"{file_name}_{value}_{i_name}_localplot.pdf",
                           dpi=1000, bbox_inches="tight")
                if png_plot:
                    fig.savefig(save_path + 
                               f"{file_name}_{value}_{i_name}_localplot.png",
                               dpi=1000, bbox_inches="tight")
                plt.close(ax.figure)


class Featureinteractionplot:
    """
    This class is designed to visualize interactions between features based on Shapley values. 

    Attributes:
        X_data : DataFrame 
        The input dataset containing feature values for the instances to be explained. 
        Each row represents an instance, and each column corresponds to a feature.

        shapley_data_all : numpy.ndarray  
        A matrix holding the Shapley interaction values for all feature pairs across the provided instances. 
        The structure should allow indexing by instance and feature pairs, 
        reflecting how each pair of features interacts in influencing the model's output. 

    Methods:
        [WILL UPDATE SOON]

    Usage Example:
        # Initialize the plotter with your data and Shapley interaction values
        interaction_plotter = FeatureInteractionPlot(X_train, shap_interactions_values)

        # Call methods to generate and display plots (methods to be implemented)
        [WILL UPDATE SOON]
    """

    def __init__(self, X_data, shapley_interaction_data_all):
        self.X_data = X_data
        self.shapley_data_all = shapley_interaction_data_all

    def plot(self, interaction=True):
        for value in list(self.shapley_data_all.keys()):
            for i_name in self.shap_indicate_feature:
                if interaction:
                    shap.dependence_plot((i_name.split('*_')[0], i_name.split('*_')[1]),
                                         np.array(self.shapley_data_all[value]),
                                         self.X_data,
                                         show=False, interaction_index=None)
                    plt.title(self.file_name + "_" + value + "_" + 'interaction_deprincipal')
                    plt.savefig(self.save_path + self.file_name + "_" + value + "_" + i_name.split('*_')[0] + 'vs_' +
                                i_name.split('*_')[1] + "_shap_interaction_depc.pdf", dpi=1000, bbox_inches="tight")
                    if self.png_plot:
                        plt.savefig(
                            self.save_path + self.file_name + "_" + value + "_" + i_name.split('*_')[0] + 'vs_' +
                            i_name.split('*_')[1] + "_shap_interaction_depc.png", dpi=1000, bbox_inches="tight")
                else:
                    shap.dependence_plot((i_name, i_name), np.array(self.shapley_data_all[value]), self.X_data,
                                         show=False,
                                         interaction_index=None)
                    plt.title(self.file_name + "_" + value + "_" + 'pf_residuals')
                    plt.savefig(self.save_path + self.file_name + "_" + value + "_" + i_name + "_shap_residuals.pdf",
                                dpi=1000, bbox_inches="tight")
                    if self.png_plot:
                        plt.savefig(
                            self.save_path + self.file_name + "_" + value + "_" + i_name + "_shap_residuals.png",
                            dpi=1000, bbox_inches="tight")
                plt.close()

    def interaction_deprincipal_plot(self, shap_indicate_feature, save_path, file_name='human_train', png_plot=True):
        self.shap_indicate_feature = shap_indicate_feature
        self.save_path = save_path
        self.file_name = file_name
        self.png_plot = png_plot
        self.plot(interaction=True)

    def principal_feature_residuals_plot(self, shap_indicate_feature, save_path, file_name='human_train',
                                         png_plot=True):
        self.shap_indicate_feature = shap_indicate_feature
        self.save_path = save_path
        self.file_name = file_name
        self.png_plot = png_plot
        self.plot(interaction=False)

    def interaction_all_plot(self, file_name, save_path, max_num_display=25):
        for name in self.shapley_data_all.keys():
            tmp = np.abs(self.shapley_data_all[name]).sum(0) / len(self.X_data.index)
            for i in range(tmp.shape[0]):
                tmp[i, i] = 0
            inds = np.argsort(-tmp.sum(0))[:max_num_display]  ##最多只展示25个特征
            tmp2 = tmp[inds, :][:, inds]
            plt.figure(figsize=(12, 12))
            plt.title(name + file_name + '_interaction')
            plt.imshow(tmp2)
            plt.yticks(range(tmp2.shape[0]), self.X_data.columns[inds], rotation=50.4, horizontalalignment="right")
            plt.xticks(range(tmp2.shape[0]), self.X_data.columns[inds], rotation=50.4, horizontalalignment="left")
            plt.gca().xaxis.tick_top()
            plt.colorbar(plt.imshow(tmp2))
            plt.savefig(save_path + file_name + "_" + name + "_" + "shap_interaction_all.pdf", dpi=1000,
                        bbox_inches="tight")
            plt.close()


class FeatureinteractionIndexsave(FeaturePlotSource):
    
    def calculate_interaction_feature_index(self):
        self.shap_interaction_feature_index = {}
        for i in self.shapley_data_all.keys():
            shap_train_interaction_feature_index = list(map(lambda x: pd.DataFrame(2*self.shapley_data_all[i][:,x,:],
                                                                           index=self.X_data.index,columns=self.X_data.columns),
                                                  range(len(self.X_data.columns))))
            shap_train_interaction_feature_index = dict(zip(list(self.X_data.columns), shap_train_interaction_feature_index)) 
            self.shap_interaction_feature_index[i] = shap_train_interaction_feature_index
    
    def save_interaction_feature_index(self,shap_indicate_feature,save_path,file_name):
        self.shap_indicate_feature=shap_indicate_feature
        for i in self.shap_interaction_feature_index.keys():
            for value in self.shap_indicate_feature:
                self.shap_interaction_feature_index[i][value].to_csv(save_path+file_name+'_'+i+'_'+value+'_shap_interaction_feature_index.csv')


def predict_probability_Fcombine(cluster_data_all, type_data_all, cluster_sum_shap, gene_ID, type_probability):
    # Initialize dictionaries to store results
    cluster_sum_shap_beyond_zero_outcome = {}
    cluster_sum_shap_less_zero_outcome = {}
    
    # Calculate cluster's shap value for proteins with values greater than zero
    for key, value in cluster_data_all.items():
        cluster_sum_shap_beyond_zero = value[value[cluster_sum_shap] > 0]
        new_df_merge = None
        for key2, value2 in type_data_all.items():
            # Merge the two data frames on the 'gene_ID' column using an inner join
            merged = pd.merge(cluster_sum_shap_beyond_zero, value2, on=gene_ID, how='inner')
            # Create a new data frame with the mean probability value, indexed by 'key2'
            new_df = pd.DataFrame({key:[merged[type_probability].mean()]},index=[key2])
            new_df_merge = pd.concat([new_df_merge, new_df], axis=0)
        # Append the new data frame to the dictionary
        cluster_sum_shap_beyond_zero_outcome[key] = new_df_merge
    # Concatenate all dataframes in the dictionary
    cluster_sum_shap_beyond_zero_outcome = reduce(lambda x,y: pd.concat([x, y], axis=1),cluster_sum_shap_beyond_zero_outcome.values())
    
    # Calculate cluster's shap value for proteins with values less than zero
    for key, value in cluster_data_all.items():
        cluster_sum_shap_less_zero = value[value[cluster_sum_shap] <= 0]
        new_df_merge = None
        for key2, value2 in type_data_all.items():
            # Merge the two data frames on the 'gene_ID' column using an inner join
            merged = pd.merge(cluster_sum_shap_less_zero, value2, on=gene_ID, how='inner')
            # Create a new data frame with the mean probability value, indexed by 'key2'
            new_df = pd.DataFrame({key:[merged[type_probability].mean()]},index=[key2])
            new_df_merge = pd.concat([new_df_merge, new_df], axis=0)
        # Append the new data frame to the dictionary
        cluster_sum_shap_less_zero_outcome[key] = new_df_merge
    # Concatenate all dataframes in the dictionary
    cluster_sum_shap_less_zero_outcome = reduce(lambda x,y: pd.concat([x, y], axis=1),cluster_sum_shap_less_zero_outcome.values())
    
    return cluster_sum_shap_beyond_zero_outcome, cluster_sum_shap_less_zero_outcome



def cluster_data_divided_feature(cluster_data_all,  cluster_indicate, cluster_sum_shap, gene_ID):

    # create a list of dataframes and reassign to cluster_data_all
    cluster_data_all_list = [cluster_data_all[x].loc[:, [gene_ID]+cluster_indicate] for x in cluster_data_all.keys()]
    cluster_data_all = dict(zip(cluster_data_all.keys(), cluster_data_all_list))
    
    # modify column names of dataframes
    for i_cluster in cluster_data_all.keys():
        for x in cluster_indicate:
            cluster_data_all[i_cluster].rename(columns={x: f"{x}_{i_cluster}"}, inplace=True)
        
        # group dataframes by columns
        grouped = cluster_data_all[i_cluster].groupby(cluster_data_all[i_cluster].columns, axis=1)

        new_dict = {}
        for col in grouped.groups:
            sub_df = grouped.get_group(col)
            new_dict[col] = sub_df
            new_dict[col]
            new_dict[col].insert(0, 'new_'+gene_ID, cluster_data_all[i_cluster][gene_ID])
            new_dict.pop(gene_ID, None)
            
            for key in new_dict.keys():
                # change the column name from 'gene_ID' to 'ID'
                new_dict[key] = new_dict[key].rename(columns={'new_'+gene_ID: gene_ID, new_dict[key].columns[-1]: cluster_sum_shap})
                
        cluster_data_all[i_cluster] = new_dict
        
    return cluster_data_all



def predict_probability_Fsingle(cluster_data_all, type_data_all, type_probability, cluster_sum_shap,feature_indicate,gene_ID):
    # Join words in a pattern
    pattern = '|'.join(feature_indicate)
    # Process data and create a new dataframe for each cluster_sum_shap_data
    def process_data(x1,key2,i_type):
        new_df = pd.DataFrame({key2:[x1.merge(type_data_all[i_type], on=gene_ID,how='inner' )[type_probability].mean()]},index=[i_type])
        return new_df
    
    
    # Filter cluster data where cluster_sum_shap is greater than or equal to zero
    cluster_sum_shap_beyond_zero = {k1: {k2: v2[v2[cluster_sum_shap] > 0] for k2, v2 in v1.items()} 
                                  for k1, v1 in cluster_data_all.items()}
    
    # Concatenate dataframes for each type and cluster_sum_shap_beyond_zero
    cluster_sum_shap_beyond_zero = {k1: {k2: pd.concat([process_data(v2,k2,i_type) for i_type in type_data_all.keys()],axis=0)  
                                      for k2, v2 in v1.items()} 
                                  for k1, v1 in cluster_sum_shap_beyond_zero.items()}
    
    # Rename keys using regex pattern
    cluster_sum_shap_beyond_zero_outcome = {k1:{re.findall(pattern,k2)[0]: v2 for k2 , v2 in v1.items()} 
                                          for k1, v1 in cluster_sum_shap_beyond_zero.items()}
    
    # Concatenate dataframes for each cluster_sum_shap
    cluster_sum_shap_beyond_zero_outcome = {k1: pd.concat([v2[k1] for k2 , v2 in cluster_sum_shap_beyond_zero_outcome.items() ],axis=1) 
                                          for i1, k1 in enumerate(feature_indicate)}

    # Filter cluster data where cluster_sum_shap is less than or equal to zero
    cluster_sum_shap_less_zero = {k1: {k2: v2[v2[cluster_sum_shap] <= 0] for k2, v2 in v1.items()} 
                                  for k1, v1 in cluster_data_all.items()}
    
    # Concatenate dataframes for each type and cluster_sum_shap_less_zero
    cluster_sum_shap_less_zero = {k1: {k2: pd.concat([process_data(v2,k2,i_type) for i_type in type_data_all.keys()],axis=0)  
                                      for k2, v2 in v1.items()} 
                                  for k1, v1 in cluster_sum_shap_less_zero.items()}
    
    # Rename keys using regex pattern
    cluster_sum_shap_less_zero_outcome = {k1:{re.findall(pattern,k2)[0]: v2 for k2 , v2 in v1.items()} 
                                          for k1, v1 in cluster_sum_shap_less_zero.items()}
    
    # Concatenate dataframes for each cluster_sum_shap
    cluster_sum_shap_less_zero_outcome = {k1: pd.concat([v2[k1] for k2 , v2 in cluster_sum_shap_less_zero_outcome.items() ],axis=1) 
                                          for i1, k1 in enumerate(feature_indicate)}
    return {'cluster_sum_shap_beyond_zero_outcome':cluster_sum_shap_beyond_zero_outcome, 'cluster_sum_shap_less_zero_outcome':cluster_sum_shap_less_zero_outcome}



    
 
