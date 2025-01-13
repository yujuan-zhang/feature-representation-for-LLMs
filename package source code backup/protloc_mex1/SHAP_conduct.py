# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:43:58 2023

@author: qq102
"""
import shap
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SHAPValueConduct:
    '''
    A class used to analyze SHAP values.
    
    Attributes:
    ----------
    explainer : SHAP explainer 
        The explainer used to calculate SHAP values, 
        for now are only support shap.TreeExplainer.
    X_input : DataFrame
        Input features.
    y_input : Series, DataFrame
        Target values.
    
    '''
    def __init__(self, explainer,X_input,y_input):
        self.explainer = explainer
        self.X_input=X_input
        self.y_input=y_input
    def shap_value_conduct(self): 
        '''
        

        Calculate the SHAP values of X_input using explainer.

        Returns
        -------
        shap_values : list
            SHAP values for each sample in X_input for each localization.

        '''
        self.shap_values = self.explainer.shap_values(self.X_input)
        return self.shap_values
    
    def save_shap_values(self, X_predict_input,shap_value_input,X_input,y_input, type_class,save_path,file_name,gene_ID):
        '''
        

        Save SHAP values, input features, target values and predictions to CSV file.  

        Parameters
        ----------
        X_predict_input : DataFrame
            Model predictions for X_input. 
        shap_value_input : list
            SHAP values for each sample in X_input for One localization.
        X_input : DataFrame
            Input features.
        y_input : Series, DataFrame
            Target values.
        type_class : str
            One target class name.               
        save_path : str
            input is file or plot save path way with no contain file name.
        file_name : str
            only indicate test or train but not specific file name, 
            such as input can't be classification_report.csv.
        gene_ID : str
            Name of the index column.

        Returns
        -------
        X_shap_save : DataFrame
            Combined information of SHAP values, inputs, targets and predictions.

        '''
                
        X_shap_save = pd.DataFrame(shap_value_input, columns=X_input.columns, index=X_input.index)
        X_shap_save = X_shap_save.join(y_input, how="inner")
        X_shap_save = X_shap_save.join(X_predict_input, how="inner")
        file_path = f"{save_path}/{file_name}_{type_class}_shap_value.csv"
        X_shap_save.to_csv(file_path, index_label=gene_ID)
        return X_shap_save
        

    def Shapley_value_save(self,X_predict_input,type_class,save_path,file_name="train",gene_ID='ID'):
        '''
        

        Save SHAP values for different classes.
        
        Parameters
        ----------
        X_predict_input : DataFrame
            Model predictions for X_input.  
        type_class : list
            Names of the target classes.                 
        save_path : str
            input is file or plot save path way with no contain file name.
        file_name : str, optional
            only indicate test or train but not specific file name, The default is "train".
        gene_ID : str, optional
            Name of the index column. The default is 'ID'.
        
        Returns
        -------
        X_shap_save : dict
            Saved information of SHAP values (SHAP values, inputs, targets and predictions) for each class.
        
        Note
        -------
        Please ensure that the type entered for the type_class parameter matches 
        the type of the model default classification,
        otherwise the results may be inaccurate!
        '''
        logging.warning('Please ensure that the type entered for the type_class parameter matches the type of the model default classification,\
                        otherwise the results may be inaccurate!')
        X_shap_save=list(map(lambda x,y:self.save_shap_values(X_predict_input,\
                                    self.shap_values[y],self.X_input,self.y_input,x,save_path,file_name,gene_ID),type_class,range(len(type_class))))
        X_shap_save=dict(zip(type_class,X_shap_save))
        return X_shap_save


class ShapInteractionIndexConduct:
    '''


    A class used to analyze SHAP interaction values.

    Attributes:
    ----------
    frac_num : float
        The fraction of samples (random select from original samples) used to calculate SHAP interaction values.
    tree_limit_num : int
        The number of trees used to calculate SHAP interaction values, suggest setting is 100.
    explainer : tree explainer
        The explainer used to calculate SHAP interaction values,
        for now are only support shap.TreeExplainer..
    shap_plot_figure_size : tuple
        The figure size of the SHAP summary interaction plot.
    plot_need : bool
        if need to draw summary interaction plot
    shap_interaction_values_storage : bool   
        Specifies whether the complete SHAP interaction values need to be stored 
        within the instance for subsequent computations. This option does not affect the output of npy files.
    shap_interaction_abs : dict
        storage the interaction index (feature interaction summing of abs value) 

    '''

    def __init__(self, frac_num, tree_limit_num, explainer, shap_plot_figure_size=(15, 10),
                 plot_need=True, shap_interaction_values_storage=False):
        self.frac_num = frac_num
        self.tree_limit_num = tree_limit_num
        self.explainer = explainer
        self.shap_plot_figure_size = shap_plot_figure_size
        self.plot_need = plot_need
        self.shap_interaction_abs = {}
        self.shap_interaction_values_storage = shap_interaction_values_storage

    def plot_shap_interaction_index_values(self, shap_interaction_values, X_input, class_names, file_name, save_path):
        '''


        Plot the SHAP summary interaction plot and save the figure.

        Parameters:
        -----------
        shap_interaction_values : list
            SHAP interaction values for each sample in X_input for one localization.
        X_input : DataFrame
            input features from randomly selected dataframe.          
        class_names : str
            Name of the target class.
        file_name : str
            only indicate test or train but not specific file name, 
            such as input can't be classification_report.csv.
        save_path : str
            input is file or plot save path way with no contain file name.


        '''

        shap.summary_plot(shap_interaction_values, X_input,
                          plot_size=self.shap_plot_figure_size, show=False, max_display=10)

        plt.savefig(save_path + file_name + "_" + class_names + "_shap_interaction.png",
                    dpi=1000, bbox_inches="tight")
        plt.savefig(save_path + file_name + "_" + class_names + "_shap_interaction.pdf",
                    dpi=1000, bbox_inches="tight")
        plt.close()

    def save_shap_interaction_index(self, shap_interaction_values, X_random, file_name, save_path, type_class):
        '''


        Save the SHAP interaction values for different classes.

        Parameters:
        -----------
        shap_interaction_values : list
            SHAP interaction values for each sample in X_random. 
        X_random : DataFrame
             input features from randomly selected dataframe.          
        file_name : str
            only indicate test or train but not specific file name, 
            such as input can't be classification_report.csv.
        save_path : str
            input is file or plot save path way with no contain file name.
        type_class : list
            Names of the target classes.


        '''
        for i, value in enumerate(type_class):
            if self.plot_need:

                self.plot_shap_interaction_index_values(shap_interaction_values[i], X_random, value, file_name,
                                                        save_path)
            else:
                print('Do not perform interaction summary plot')
            shap_interaction_data = pd.DataFrame(shap_interaction_values[i][0, :, :],
                                                 index=X_random.columns, columns=X_random.columns)
            shap_interaction_data['feature_max'] = shap_interaction_data.apply(lambda x: sorted(abs(x))[-2],
                                                                               axis='columns')
            shap_interaction_data.to_csv(save_path + file_name + "_" + value + "_dim0_shap_interaction.csv",
                                         index_label='feature_name')

            shap_interaction_abs = pd.DataFrame(np.abs(shap_interaction_values[i]).sum(0),
                                                index=X_random.columns, columns=X_random.columns)
            shap_interaction_abs.to_csv(save_path + file_name + "_" + value + "_shap_interaction_abs.csv",
                                        index_label='feature_name')
            self.shap_interaction_abs[value] = shap_interaction_abs

        X_random.to_csv(save_path + file_name + "_feature_random.csv", index_label='ID')

        if self.shap_interaction_values_storage:

            self.shap_interaction_values_all = dict(zip(type_class, shap_interaction_values))
            np.save(save_path + file_name + "_shap_interaction_all.npy", self.shap_interaction_values_all)
        else:
            shap_interaction_values_all = dict(zip(type_class, shap_interaction_values))
            np.save(save_path + file_name + "_shap_interaction_all.npy", shap_interaction_values_all)

    def conduct(self, X_input, file_name, save_path, type_class):
        '''


        Conduct the analysis of SHAP interaction values.

        Parameters:
        -----------
        X_input : DataFrame        
             input features from full dataframe.          
        file_name : str
            only indicate test or train but not specific file name, 
            such as input can't be classification_report.csv.
        save_path : str
            input is file or plot save path way with no contain file name.
        type_class : list
            Names of the target classes.  

        Output:
        --------
        SHAP summary interaction plot (PNG and PDF format)
        SHAP interaction strength data (CSV format), Note: calculate format are displayed in <under review article>   
        Features data randomly selected from original dataframe (CSV format)
        SHAP interaction values for all classes (NPY format)

        Note
        -------
        Please ensure that the type entered for the type_class parameter matches 
        the type of the model default classification,
        otherwise the results may be inaccurate!
        '''
        logging.warning('Please ensure that the type entered for the type_class parameter matches the type of the model default classification,\
                        otherwise the results may be inaccurate!')

        X_random = X_input.sample(frac=self.frac_num, replace=False, random_state=0, axis=0)
        shap_interaction_values = self.explainer.shap_interaction_values(X_random, tree_limit=self.tree_limit_num)

        self.save_shap_interaction_index(shap_interaction_values, X_random, file_name, save_path, type_class)



class DeepSHAPValueConduct:
    '''
    A class used to analyze Deep SHAP values.
    
    Attributes:
    ----------
    explainer : SHAP DeepExplainer 
        The explainer used to calculate SHAP values.
    X_input : DataFrame
        Input features.
    y_input : Series, DataFrame
        Target values.
    '''
    def __init__(self, explainer , X_input, X_input_tensor,y_input):
        
        
        # Randomly select samples from X_input to form the background dataset
        
        self.explainer = explainer 
        self.X_input = X_input
        self.X_input_tensor=X_input_tensor
        self.y_input = y_input
    
    def shap_value_conduct(self):
        '''
        Calculate the SHAP values of X_input using explainer.

        Returns
        -------
        shap_values : list
            SHAP values for each sample in X_input for each localization.
        '''
        self.shap_values = self.explainer.shap_values(self.X_input_tensor)  # Convert DataFrame to numpy array
        return self.shap_values

    def save_shap_values(self, X_predict_input,shap_value_input,X_input,y_input, type_class,save_path,file_name,gene_ID):
        '''
        

        Save SHAP values, input features, target values and predictions to CSV file.  

        Parameters
        ----------
        X_predict_input : DataFrame
            Model predictions for X_input. 
        shap_value_input : list
            SHAP values for each sample in X_input for One localization.
        X_input : DataFrame
            Input features.
        y_input : Series, DataFrame
            Target values.
        type_class : str
            One target class name.               
        save_path : str
            input is file or plot save path way with no contain file name.
        file_name : str
            only indicate test or train but not specific file name, 
            such as input can't be classification_report.csv.
        gene_ID : str
            Name of the index column.

        Returns
        -------
        X_shap_save : DataFrame
            Combined information of SHAP values, inputs, targets and predictions.

        '''
                
        X_shap_save = pd.DataFrame(shap_value_input, columns=X_input.columns, index=X_input.index)
        X_shap_save = X_shap_save.join(y_input, how="inner")
        X_shap_save = X_shap_save.join(X_predict_input, how="inner")
        file_path = f"{save_path}/{file_name}_{type_class}_shap_value.csv"
        X_shap_save.to_csv(file_path, index_label=gene_ID)
        return X_shap_save
        

    def Shapley_value_save(self,X_predict_input,type_class,save_path,file_name="train",gene_ID='ID'):
        '''
        

        Save SHAP values for different classes.
        
        Parameters
        ----------
        X_predict_input : DataFrame
            Model predictions for X_input.  
        type_class : list
            Names of the target classes.                 
        save_path : str
            input is file or plot save path way with no contain file name.
        file_name : str, optional
            only indicate test or train but not specific file name, The default is "train".
        gene_ID : str, optional
            Name of the index column. The default is 'ID'.
        
        Returns
        -------
        X_shap_save : dict
            Saved information of SHAP values (SHAP values, inputs, targets and predictions) for each class.
        
        Note
        -------
        Please ensure that the type entered for the type_class parameter matches 
        the type of the model default classification,
        otherwise the results may be inaccurate!
        '''
        logging.warning('Please ensure that the type entered for the type_class parameter matches the type of the model default classification,\
                        otherwise the results may be inaccurate!')
        X_shap_save=list(map(lambda x,y:self.save_shap_values(X_predict_input,\
                                    self.shap_values[y],self.X_input,self.y_input,x,save_path,file_name,gene_ID),type_class,range(len(type_class))))
        X_shap_save=dict(zip(type_class,X_shap_save))
        return X_shap_save


class EBMFeatAggregValueConduct:
    '''
    A class used to calculate EBM feature aggregation values.

    Attributes:
    ----------
    explainer : EBM model 
        The explainer used to calculate EBM feature aggregation values, 
        for now are only support EBM model(interpret.glassbox.ExplainableBoostingClassifier).
    X_input : DataFrame
        Input features.
    y_input : Series, DataFrame
        Target values.

    '''

    def __init__(self, explainer, X_input, y_input):
        self.explainer = explainer
        self.X_input = X_input
        self.y_input = y_input
        # self.class_scores = {}

    def FeatAggreg_value_conduct(self):
        '''

        Calculate the feature aggregation values of X_input using explainer.

        Returns
        -------
        fearure_aggregation_value_Array : dict of Array
        EBM feature aggregation values for each sample in X_input for each localization
        with a form of Array.
        fearure_aggregation_value_Dataframe : dict of dataframe
        EBM feature aggregation values for each sample in X_input for each localization
        with a form of dataframe.

        '''
        ebm_local = self.explainer.explain_local(self.X_input, self.y_input)

        num_classes = len(ebm_local.data(0)['scores'][0])

        num_features = len(ebm_local.data(0)['names'])
        num_instances = len(self.X_input)

        # Initialize an array to hold all feature scores for all classes
        # Shape will be (num_classes, num_features, num_instances)
        all_scores = np.zeros((num_classes, num_instances, num_features))

        for i in range(num_instances):
            explanation = ebm_local.data(i)
            for feature_index in range(num_features):
                # This assumes that the scores are in the same order as the feature names
                all_scores[:, i, feature_index] = explanation['scores'][feature_index]

        # Now 'all_scores' is a 3D array where the axes are [class, feature, instance]
        # To split by cell type (class), simply index 'all_scores' by class
        self.fearure_aggregation_value_Array = {class_index: all_scores[class_index, :, :] for class_index in
                                                range(num_classes)}

        ##将数组修改为矩阵
        self.fearure_aggregation_value_Dataframe = {}

        for class_index in range(num_classes):
            # The index is set to the index of X_train
            # The columns are set to the feature names
            self.fearure_aggregation_value_Dataframe[class_index] = pd.DataFrame(
                self.fearure_aggregation_value_Array[class_index],
                index=self.X_input.index,
                columns=self.X_input.columns
            )

        return self.fearure_aggregation_value_Array, self.fearure_aggregation_value_Dataframe

    def FeatAggreg_value_save(self, save_path, file_name="train", gene_ID='ID'):
        """
        Save fearure_aggregation_value for different localization.

        Parameters
        ----------

        save_path : str
            input is file or plot save path way with no contain file name.
        file_name : str, optional
            only indicate test or train but not specific file name, The default is "train".
        gene_ID : str, optional
            Name of the index column. The default is 'ID'.

        """

        for class_index, class_name in enumerate(self.explainer.classes_):
            # X_shap_save = pd.DataFrame(shap_value_input, columns=X_input.columns, index=X_input.index)
            X_shap_save = self.fearure_aggregation_value_Dataframe[class_index].join(self.y_input, how="inner")
            X_shap_save = X_shap_save.join(
                pd.Series(self.explainer.predict(self.X_input), index=self.X_input.index, name='predict'), how="inner")
            file_path = f"{save_path}/{file_name}_{class_name}_shap_value.csv"
            X_shap_save.to_csv(file_path, index_label=gene_ID)

    