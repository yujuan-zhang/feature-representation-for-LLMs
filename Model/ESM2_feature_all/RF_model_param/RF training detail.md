### RF model training

RF model training incorporates various feature sets including 'cls', 'eos', 'pho', 'VAE_cls', 'segment0', 'AA_doc2vec', and 'feature all'. Each feature set is used to individually train an RF model. Specifically for the 'feature all' set, after combining all available features, feature selection is conducted based on their importance via the `sklearn.feature_selection` module during hyperparameter optimization using the `optuna` package. Ultimately, only 3152 features are retained. The specific training steps are as follows:

1. For features except 'feature all', feature augmentation is initially performed using the `imblearn` package, with the particular algorithm details elaborated in [DNN training detrail](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/ESM2_feature_all/DNN_model_param/DNN%20MLP.md
   ). Note that the features for the training set, test set, or inference data must be standardized as previously described.

   ```python
   from imblearn.over_sampling import SMOTE
   import os
   import pandas as pd
   import numpy as np
   cancer_name = 'human'
   X_train_scale = 'Assume this is your 1280-dimensional, standardized CLS training data.'
   y_train = 'Assume this is your CLS training data label'
   sampling_strategy = {"Nucleus": 1847, "Cytoplasm": 1500, "Cell membrane": 1500,
                       "Secreted": 1500, "Mitochondrion": 1500, "Endoplasmic reticulum": 1500,
                       "Golgi apparatus": 1500, "Cell projection": 1500, "Lysosome" : 1500,
                        "Cell junction" : 1500 }
   
   smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
   
   # Oversampling is performed on the training data 
   X_train_res, y_train_res = smote.fit_resample(X_train_scale, y_train)
   ```

2. We employ the `optuna` package to conduct 10-fold `StratifiedKFold` cross-grid search for optimal hyperparameters. The objective for hyperparameter optimization is to maximize the average F1-score across all folds. The hyperparameters under consideration include the number of trees in the random forest `n_estimators`(ranging from 50 to 200), the maximum depth of the trees `max_depth` (ranging from 1 to 30), the number of features randomly chosen at each trees `max_features` ('sqrt' or 'log2'), and the minimum number of samples required to further split an internal node `min_samples_split` (ranging from 2 to 50).

   For details on `optuna` optimization techniques such as early pruning, please refer to [VAE training detail](https://github.com/yujuan-zhang/feature-representation-for-LLMs/blob/main/Model/VAE%20model/Res_VAE%20training%20detail.md
   ).

   ```python
   
   import optuna
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.pipeline import Pipeline
   from sklearn.model_selection import StratifiedKFold
   from sklearn.metrics import f1_score
   from sklearn.model_selection import cross_val_score
   
   cv_test = 'f1_weighted'
   k_cv_num = 10
   
   if os.path.isdir(save_path+'/RF'):
     pass
   else:
     os.makedirs(save_path+'/RF')
   
   # define objective function
   def objective(trial):
       # Create Model Instance (Random Forest) - During the optimization process
       forest = RandomForestClassifier(
           n_estimators=trial.suggest_int('n_estimators', 50, 200),
           class_weight='balanced',
           max_depth=trial.suggest_int('max_depth', 1, 30),
           max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
           min_samples_split=trial.suggest_int('min_samples_split', 2, 50),
           random_state=0)
   
       pipe_param = Pipeline([("Model", forest)])
       kfold = StratifiedKFold(n_splits=k_cv_num, shuffle=True, random_state=0)
       
       scores = cross_val_score(pipe_param, X_train_res, y_train_res.squeeze(), cv=kfold, scoring=cv_test, n_jobs=-1)
       
       return scores.mean()  # we aim to maximize the F1 score.
   
   study = optuna.create_study(direction='maximize')  # we aim to maximize the F1 score.
   study.optimize(objective, n_trials=50)  # run 50 trail
   
   # get the optimal hyperparameters 
   best_params = study.best_params
   best_params = pd.DataFrame(best_params, index=['RF_CLS'])
   # save the optimal hyperparameters
   best_params.to_csv('your save path')
   ```

3. Reconstruct and train the model based on the optimal hyperparameters `best_params`

   ```python
   forest_number = best_params.loc['RF_CLS', 'n_estimators']
   forest_max_depth = best_params.loc['RF_CLS', 'max_depth']
   forest_max_features = best_params.loc['RF_CLS', 'max_features']
   forest_min_samples_split = best_params.loc['RF_CLS', 'min_samples_split']
   forest_class_weight = 'balanced'
   forest=RandomForestClassifier(max_depth=forest_max_depth, max_features=forest_max_features,
                                     random_state=0, n_estimators=forest_number, 
                                     class_weight=forest_class_weight, min_samples_split=forest_min_samples_split)
   kfold=StratifiedKFold(n_splits=kfold_num,shuffle=True,random_state=0) # using StratifiedKFold 
   scores=cross_val_score(forest, X_train_res, y_train_res, scoring=cv_test, cv=kfold, n_jobs=-1) 
   np.mean(scores)
   scores={cancer_name:scores}
   scores=pd.DataFrame(scores, columns=[cancer_name])
   
   # fit model 
   forest.fit(X_train_res, y_train_res)
       
   # Output the trained model along with the test and training sets.
   '''
   Obtain the version of scikit-learn (the saved model will be compatible with this version of scikit-learn, and  the same applies when loading it later).
   '''
   scikit_version=sklearn.__version__ 
   #Save the model as a pickle file.
   joblib.dump(forest,'your save path'+'ESM2_cls'+cancer_name+scikit_version+".pkl")
   ```

   

4. For the 'feature all' attributes, the training process is very similar, with the only difference being the need to first use the `SelectFromModel` function from the `sklearn.feature_selection` module to perform feature selection based on feature importance. Specifically, the feature importance threshold `select_model_threshold` is calculated as `0.75 * mean`, which will reduce the dimensionality of ‘feature_all’ from 6418 to 3152. This process is described as building an RF_filter model for feature dimensionality reduction. Specifically, in our training experiment, you may not achieve this importance threshold due to not following our recommended experimental environment. However, it's not a significant issue because we hope our observation is not dependent on the downstream model but aims for more robustness. We will continue to research the feature representation preference.

   ```python
   X_train_scale = 'Assume this is your 6418-dimensional, standardized all feature training data.'
   y_train = 'Assume this is your CLS training data label'
   sampling_strategy = {"Nucleus": 1847, "Cytoplasm": 1500, "Cell membrane": 1500,
                       "Secreted": 1500, "Mitochondrion": 1500, "Endoplasmic reticulum": 1500,
                       "Golgi apparatus": 1500, "Cell projection": 1500, "Lysosome" : 1500,
                        "Cell junction" : 1500 }
   
   smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
   
   # Oversampling is performed on the training data 
   X_train_res, y_train_res = smote.fit_resample(X_train_scale, y_train) 
   
   from sklearn.feature_selection import SelectFromModel
   
   # define objective function
   def objective(trial):
       
       model_select = SelectFromModel(estimator=RandomForestClassifier(n_estimators=trial.suggest_int('Feature selection_n_estimators', 50, 200), 
                      class_weight='balanced', random_state=0), 
                      threshold=trial.suggest_categorical('Feature selection_threshold', [0, "median", "mean", "1.5*mean", "1.25*mean", "0.5*mean", "0.75*mean"]))
   
       # # Create Model Instance (Random Forest) - During the optimization process
       forest = RandomForestClassifier(
           n_estimators=trial.suggest_int('n_estimators', 50, 200),
           class_weight='balanced',
           max_depth=trial.suggest_int('max_depth', 1, 30),
           max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
           min_samples_split=trial.suggest_int('min_samples_split', 2, 50),
           random_state=0)
       
       pipe_param = Pipeline([("Feature selection", model_select), ("Model", forest)])
       kfold = StratifiedKFold(n_splits=k_cv_num, shuffle=True, random_state=0)
       
       scores = cross_val_score(pipe_param, X_train_res, y_train_res.squeeze(), cv=kfold, scoring=cv_test, n_jobs=-1)
       
       return scores.mean()  # we aim to maximize the F1 score.
   
   study = optuna.create_study(direction='maximize')  # we aim to maximize the F1 score.
   study.optimize(objective, n_trials=30)  # run 50 trail
   
   # get the optimal hyperparameters 
   best_params = study.best_params
   best_params = pd.DataFrame(best_params, index=['feature all'])
   
   
   
   forest_number = best_params.loc['feature all', 'n_estimators']
   select_model_threshold = best_params.loc['feature all', 'Feature selection_threshold']
   forest_max_depth = best_params.loc['feature all', 'max_depth']
   forest_max_features = best_params.loc['feature all', 'max_features']
   forest_min_samples_split = best_params.loc['feature all', 'min_samples_split']
   model_select_number = best_params.loc['feature all', 'Feature selection_n_estimators']
   
   
   # Add Feature Selection Step
   model_select = SelectFromModel(estimator=RandomForestClassifier(n_estimators=model_select_number, 
                      class_weight='balanced', random_state=0), 
                      threshold=select_model_threshold)
       
   # Create Model Instance (Random Forest) - During the optimization process
   forest=RandomForestClassifier(max_depth=forest_max_depth, max_features=forest_max_features,
                                     random_state=0, n_estimators=forest_number, 
                                     class_weight=forest_class_weight, min_samples_split=forest_min_samples_split)
   
   pipe_param = Pipeline([("Feature selection", model_select), ("Model", forest)])
   kfold=StratifiedKFold(n_splits=kfold_num,shuffle=True,random_state=0) # 分类用StratifiedKFold，回归用Kfold
   scores=cross_val_score(pipe_param, X_train_res, y_train_res, scoring=cv_test, cv=kfold, n_jobs=-1) 
   np.mean(scores)
   scores={cancer_name:scores}
   scores=pd.DataFrame(scores, columns=[cancer_name])
   
   # fit model
   pipe_param.fit(X_train_res, y_train_res)
       
   forest_save_data={"training set":["{:.3f}".format(pipe_param.score(X_train_scale,y_train))],
                      "testing set":["{:.3f}".format(pipe_param.score(X_test_scale,y_test))],
                      "feature_number":["{:}".format(X_train_scale.shape[1])]}
   forest_save_dataframe=pd.DataFrame(forest_save_data,
                                           columns=["training set","testing set","feature_number"],
                                           index=[cancer_name])
   
   # Output the trained model along with the test and training sets.
   '''
   Obtain the version of scikit-learn (the saved model will be compatible with this version of scikit-learn, and  the same applies when loading it later).
   '''
   scikit_version=sklearn.__version__ 
   #Save the model as a pickle file.
   joblib.dump(pipe_param.named_steps['Model'],'your save path'+'ESM2_feature all'+ prefix+cancer_name+scikit_version+".pkl")
   
   # Obtain the boolean mask for supported features
   support = pipe_param.named_steps['Feature selection'].get_support()
   # Get the column names of the original dataset
   original_columns = X_train_res.columns
   # Use the mask to obtain the selected column names
   selected_columns = original_columns[support]
   
   # Index the original DataFrame using the mask or column names 
   X_train_scale = pd.DataFrame(pipe_param.named_steps['Feature selection'].transform(X_train_scale), columns=selected_columns)
      
   X_train_dataframe=X_train_scale.join(y_train,how="inner")
   X_train_dataframe.to_csv('your save path'+'ESM2_feature all'+"train_"+" forest_train_dataframe.csv",index_label='ID')
   
   ```

   



