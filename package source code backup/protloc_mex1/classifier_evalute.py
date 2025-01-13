

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
import logging

from packaging import version
import warnings
import seaborn as sns
try:
    import sklearn
    if version.parse(sklearn.__version__) < version.parse('1.0.2'):
        warnings.warn("Your sklearn version is older than 1.0.2 and may not operate correctly.")
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    from sklearn.metrics import matthews_corrcoef
except ImportError:
    warnings.warn("Sklearn not found. Some functions will not be available.")

# try:
#     import mglearn
# except ImportError:
#     warnings.warn("Mglearn not found. Some functions will not be available.")

    
class ClassifierEvaluator:
    """
    A class for evaluating the performance of a classifier model.
    
    Attributes:
    ----------
        model_prob  : np.array
            input is model.predict_proba(X_data)
        y_data : np.array or pd.Series, pd.DataFrame
            contain the true labels of the samples.
        X_hat_data : pd.DataFrame
            A Pandas DataFrame containing the predicted class labels for each sample with columns is predict.
        type_class : np.array
            A array of the class labels and must as the same order as type of the model default classification.
        
    """
    def __init__(self, model_prob, y_data,X_hat_data,type_class):
        self.model_prob = model_prob
        self.y_data = np.array(y_data)
        self.X_hat_data=X_hat_data
        self.type_class=type_class
        logging.warning('Please ensure that the type entered for the type_class parameter matches the type of the model default classification,\
                        otherwise the results may be inaccurate!')
    class ClassificationReportConduct:
        def __init__(self, func):
            self.func = func
            self.self_obj = None
        
        def __get__(self, instance, owner):
            self.self_obj = instance
            return self
        
        def __call__(self, *args, **kwargs):
            self.plot(*args, **kwargs)
        
        def accuracy_calculate(self,protein_class):
            
            type_mapping={}
            for value in self.self_obj.type_class:
                if value!=protein_class:
                    type_mapping[value]='other'
                else:
                    type_mapping[value]=value
            y_true=pd.Series(self.self_obj.y_data).map(type_mapping)  #Sample actual values
            y_pred=self.self_obj.X_hat_data["predict"].map(type_mapping) #Model Predicted Values
            accuracy=accuracy_score(y_true,y_pred)
            accuracy_save_data=pd.DataFrame({protein_class:["{:.5f}".format(accuracy)]},columns=[protein_class]
                                            ,index=['accuracy'])
            return(accuracy_save_data)
        
        def auc_calculate(self,protein_class):
           
            type_mapping={}
            for value in self.self_obj.type_class:
                if value!=protein_class:
                    type_mapping[value]=0
                else:
                    type_mapping[value]=1
            y_true=pd.Series(self.self_obj.y_data).map(type_mapping)  #Sample actual values
            y_pred=self.self_obj.X_hat_data["predict"].map(type_mapping) #Model Predicted Values
            auc=roc_auc_score(y_true,y_pred)
            auc_save_data=pd.DataFrame({protein_class:["{:.5f}".format(auc)]},columns=[protein_class]
                                            ,index=['auc'])
            return(auc_save_data)
        
        def mcc_calculate(self, protein_class):
            type_mapping = {}
            for value in self.self_obj.type_class:
                if value != protein_class:
                    type_mapping[value] = 0
                else:
                    type_mapping[value] = 1
            y_true = pd.Series(self.self_obj.y_data).map(type_mapping)  # Sample actual values
            y_pred = self.self_obj.X_hat_data["predict"].map(type_mapping) # Model Predicted Values
            mcc = matthews_corrcoef(y_true, y_pred)
            mcc_save_data = pd.DataFrame({protein_class: ["{:.5f}".format(mcc)]}, columns=[protein_class], index=['mcc'])
            return mcc_save_data
    
        
        def plot(self,save_path,file_name):
            
            ##accuracy
            accuracy_save_list=list(map(lambda x:self.accuracy_calculate(x),self.self_obj.type_class))
            accuracy_save_data=reduce(lambda x,y:pd.concat([x,y],axis=1),accuracy_save_list)
            ##auc
            auc_save_list=list(map(lambda x:self.auc_calculate(x),self.self_obj.type_class))
            auc_save_data=reduce(lambda x,y:pd.concat([x,y],axis=1),auc_save_list)
            ##mcc
            mcc_save_list = list(map(lambda x: self.mcc_calculate(x), self.self_obj.type_class))
            mcc_save_data = reduce(lambda x, y: pd.concat([x, y], axis=1), mcc_save_list)

            ## Merging and save accuracy and auc
            accuracy_auc_mcc_outcome=pd.concat([accuracy_save_data,auc_save_data,mcc_save_data],axis=0)
            accuracy_auc_mcc_outcome.to_csv(save_path+file_name+'_classification_report2.csv',index_label='index')
            ## Save f1, recall, precision
            test_class_report=classification_report(self.self_obj.y_data, 
                                                    self.self_obj.X_hat_data["predict"].to_numpy(),
                                                    output_dict=True)
            ## Save test set results
            test_class_report_save=dict()
            for i in self.self_obj.type_class.tolist():
                test_class_report_save[i]=test_class_report.get(i)
            for i in test_class_report_save.keys():
                test_class_report_save[i]=pd.Series(test_class_report_save[i],name=i)
            test_class_report_save=pd.DataFrame(reduce(lambda x,y:pd.concat([x,y],axis=1),test_class_report_save.values()))
            test_class_report_save.to_csv(save_path+file_name+'_classification_report.csv',index_label='index')
    @ClassificationReportConduct
    def classification_report_conduct(self,save_path,file_name, *args, **kwargs):
        """
        Parameters
        ----------
        save_path : str
            input is file or plot save path way with no contain file name.
        file_name : str
            only indicate test or train but not specific file name, 
            such as input can't be classification_report.csv.
        *args : NoneType
            Placeholder parameter, no real meaning for now.
        **kwargs : NoneType
            Placeholder parameter, no real meaning for now.

        Returns
        -------
        Export the files of evaluation metrics such as f1-score of the classification 
        to the specified path.
        output : _classification_report.csv and _classification_report2.csv
        
        """
        pass

            


    class ClassificationEvaluatePlot:
        def __init__(self, func):
            self.func = func
            self.self_obj = None
        
        def __get__(self, instance, owner):
            self.self_obj = instance
            return self
        
        def __call__(self, *args, **kwargs):
            self.plot(*args, **kwargs)
        
        def two_features_roc(self):
            
            type_mapping={}
            type_mapping = {self.self_obj.type_class[0]: 0,
                                 self.self_obj.type_class[1]: 1}
            y_true = pd.Series(self.self_obj.y_data).map(type_mapping)
            # calculate target probabilities
            target_probabilities = self.self_obj.model_prob[:,1]
            # compute false positive rate, true positive rate, and threshold
            false_positive_rate, true_posistive_rate, threshold = roc_curve(y_true, target_probabilities)
            # find threshold closest to 0.5 for RF (different for other classifiers)
            close_default_rf = np.argmin(np.abs(threshold - 0.5))
            # compute ROC AUC
            roc_auc = auc(false_positive_rate, true_posistive_rate)
            # plot ROC curve
            plt.subplots(figsize=(7, 5.5))
            plt.plot(false_positive_rate, true_posistive_rate, color='darkorange', lw=2,
                     label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.plot(false_positive_rate[close_default_rf], true_posistive_rate[close_default_rf], 'o',
                     markersize=10, label="threshold 0.5RF", fillstyle="none", c='k', mew=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            
        # def confusion_matrix_plot(self,figsize):
            
        #     # create confusion matrix
        #     matrix = confusion_matrix(self.self_obj.y_data, self.self_obj.X_hat_data["predict"].to_numpy())
        #     plt.figure(figsize=figsize)
        #     mglearn.tools.heatmap(matrix, xlabel='Predicted label', ylabel='True label',
        #                           xticklabels=np.array(pd.Series(self.self_obj.type_class).str.slice(0,2)),
        #                           yticklabels=self.self_obj.type_class, cmap=plt.cm.viridis_r, fmt="%d")
        #     plt.gca().invert_yaxis()
        def confusion_matrix_plot(self, figsize):
            # create confusion matrix
            matrix = confusion_matrix(self.self_obj.y_data, self.self_obj.X_hat_data["predict"].to_numpy())
            
            # Assuming type_class and x and y labels
            xticklabels = np.array(pd.Series(self.self_obj.type_class).str.slice(0,2))
            yticklabels = self.self_obj.type_class

            # Set figure size
            plt.figure(figsize=figsize)

            # Use seaborn
            sns.heatmap(matrix, annot=True, cmap='viridis_r', fmt='d', xticklabels=xticklabels, yticklabels=yticklabels)

            # Configure the plot details
            plt.gca().invert_yaxis()
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
    
        def confusion_matrix_plot_percentage(self, figsize):
            # create confusion matrix
            matrix = confusion_matrix(self.self_obj.y_data, self.self_obj.X_hat_data["predict"].to_numpy())
            
            # Convert to percentage
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

            # Assuming type_class and x and y labels
            xticklabels = np.array(pd.Series(self.self_obj.type_class).str.slice(0,2))
            yticklabels = self.self_obj.type_class

            # Set figure size
            plt.figure(figsize=figsize)

            # Use seaborn
            sns.heatmap(matrix, annot=True, cmap='viridis_r', fmt='.2%', xticklabels=xticklabels, yticklabels=yticklabels)

            # Configure the plot details
            plt.gca().invert_yaxis()
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
    

        def plot(self,save_path,file_name,figsize):
            
            if len(pd.unique(self.self_obj.y_data)) <= 2:
                self.two_features_roc()
                plt.title(file_name+'ROC curve')
                plt.savefig(save_path+file_name+'_roc.png',dpi=1000,bbox_inches="tight")
                plt.savefig(save_path+file_name+'_roc.pdf',dpi=1000,bbox_inches="tight")
                plt.close()
                self.confusion_matrix_plot(figsize)
                plt.title(file_name+'Confusion matrix')
                plt.savefig(save_path+file_name+'_confusion_matrix.png',dpi=1000,bbox_inches="tight")
                plt.savefig(save_path+file_name+'_confusion_matrix.pdf',dpi=1000,bbox_inches="tight")
                plt.close()
                self.confusion_matrix_plot_percentage(figsize)
                plt.title(file_name+'Confusion matrix percentage')
                plt.savefig(save_path+file_name+'_confusion_matrix_percentage.png',dpi=1000,bbox_inches="tight")
                plt.savefig(save_path+file_name+'_confusion_matrix_percentage.pdf',dpi=1000,bbox_inches="tight")
                plt.close()
                
            else:
                self.confusion_matrix_plot(figsize)
                plt.title(file_name+'Confusion matrix')
                plt.savefig(save_path+file_name+'_confusion_matrix.png',dpi=1000,bbox_inches="tight")
                plt.savefig(save_path+file_name+'_confusion_matrix.pdf',dpi=1000,bbox_inches="tight")
                plt.close()
                self.confusion_matrix_plot_percentage(figsize)
                plt.title(file_name+'Confusion matrix percentage')
                plt.savefig(save_path+file_name+'_confusion_matrix_percentage.png',dpi=1000,bbox_inches="tight")
                plt.savefig(save_path+file_name+'_confusion_matrix_percentage.pdf',dpi=1000,bbox_inches="tight")
                plt.close()
    @ClassificationEvaluatePlot
    def classification_evaluate_plot(self,save_path,file_name,figsize, *args, **kwargs):
        """
        Parameters
        ----------
        save_path : str
            input is file or plot save path way with no contain file name.
        file_name : str
            only indicate test or train but not specific file name, 
            such as input can't be classification_report.csv.
        figsize : tuple
            set the pdf size of output images
        *args : NoneType
            Placeholder parameter, no real meaning for now.
        **kwargs : NoneType
            Placeholder parameter, no real meaning for now.

        Returns
        -------
        Export the files of evaluation metrics such as f1-score of the classification 
        to the specified path.
        output : _classification_report.csv and _classification_report2.csv
        
        """
        pass
      