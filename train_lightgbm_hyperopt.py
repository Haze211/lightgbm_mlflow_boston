import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class LGBHyperoptProd(object):
    def __init__(self):
        pass
    def prepare_lgb_dataset(self):
        X, y = load_boston(return_X_y=True)
        boston_df = pd.DataFrame(X)
        boston_df['y'] = y

        boston_df.columns = ['crim', 'zn', 'indus', 
        'chas', 'nox', 'rm', 'age', 'dis', 'rad', 
        'tax', 'ptratio', 'b', 'lstat', 'medv']

        X = boston_df.drop('medv', axis = 1)
        y = boston_df['medv']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)

        #constructing scaler on training data. this way we dont promote data leak.
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)    

        ### utilizing lgb api
        train_data = lgb.Dataset(X_train_scaled, label=y_train, free_raw_data=False)
        validation_data  = lgb.Dataset(X_test_scaled, label = y_test, free_raw_data=False)

        return train_data, validation_data, scaler
        
    def get_optim_space(self, user_space:dict=False):

        #setting param grid for hyperopt. for whole list of params visit: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#weight-data
        # also a good doc to check which parameters influence speed, quality of the model and overfitting https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

        defined_space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            'num_boost_round': hp.quniform('num_boost_round', 50, 500, 10),
            'num_leaves': hp.quniform('num_leaves', 1, 11, 1), #for this dataset setting high number of leaves (> 31) is pointless
            'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
            'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
            #'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart'])
        }

        if user_space:
            return user_space
        else:
            return space        

    def get_optim_objective(self, train_data):
        def objective(params:dict, folds:int = 3):
            
            params['num_boost_round'] = int(params['num_boost_round'])
            params['num_leaves'] = int(params['num_leaves'])
            params['seed'] = 18
            params['verbose'] = -1

            cv_result = lgb.cv(
                params,
                train_data,
                num_boost_round=params['num_boost_round'],
                metrics=['l1', 'l2', 'mape'],
                nfold=3,
                #set starified=False ot use cv for regression
                stratified=False,
                verbose_eval=10,
                early_stopping_rounds=folds)    
            loss = cv_result['mape-mean'][-1]
            return loss
        return objective
        
    def get_optimal_model_hyperparams(self):
        pass
    def tag_model_for_production(self, experiment_name):
        #setup mlflow experiment. log model with best 
        #hyperparams and tag it with lable
        pass
    def serve_model(self):
        #execute as bash command
        pass
    def get_preds(self, data, port=1234):
        pass