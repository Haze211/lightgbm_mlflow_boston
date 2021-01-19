import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import subprocess
import requests
import warnings


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class LGBHyperoptProd(object):
    def __init__(self, work_dir):
        self.PATH = work_dir
        
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

            cv_result = lgb.cv(
                params,
                train_data,
                num_boost_round=params['num_boost_round'],
                metrics=['l1', 'l2', 'mape'],
                nfold=folds,
                #set starified=False ot use cv for regression
                stratified=False,
                verbose_eval=10,
                early_stopping_rounds=10)    
            loss = cv_result['mape-mean'][-1]
            return loss
        return objective
        
    def get_optimal_model_hyperparams(self, params, train_data, model_tag):
        best = fmin(fn=self.get_optim_objective,
                space=params,
                algo=tpe.suggest,
                max_evals=maxevals,
                trials=trials)
        return best
    def tag_model_for_production(self, experiment_name, train_data, hyperparams):
        ##ToDO: add input and output schema with mlflow.lightgbm.signature 
        
        #Set MLflow experiment name. Used for better tracking structure
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            mlflow.set_experiment(experiment_name)
        #setup mlflow experiment. run and log model with best 
        #hyperparams and tag it with lable

        X = train_data.get_data()
        y = train_data.get_label()

        #change type from float to int, recuired by lgb
        params['num_boost_round'] = int(params['num_boost_round'])
        params['num_leaves'] = int(params['num_leaves'])

        with mlflow.start_run() as run:
            ### unpack dict with best params
            model = lgb.LGBMRegressor(**best)
            model.fit(X,y)
            #best item is dict with best hyperparams returned from minimizing objective (mape)
            for name, value in best.items():
                mlflow.log_param(name, value)
                mlflow.log_metric('mape', trials.best_trial['result']['loss'])
                mlflow.sklearn.log_model(model, "model")
                #tag model for easier search across possible big model pool in mlflow.
                #idealy, there should be 1 prod model per dataset to avoid confusion.
                mlflow.set_tag("model.type", model_tag)
                mlflow.set_tag("dataset_name", "boston")
                #dataset version is used to distinguish possbile diffrenet preprocessing options.
                #might be used together with dvc workflow
                mlflow.set_tag("dataset_version", "boston")

    def load_prod_model(self, model_tag, run_id=None):
        runs = mlflow.search_runs()
        runs=runs.loc[runs['status']=='FINISHED']
        if run_id:
            model_artifact = runs.loc[runs['run_id']==run_id]['artifact_uri'][0]
        else:
            #logging several runs with same tag will create inconsistency. first row (latest run) is returned.
            model_artifact = runs.loc[(runs['tags.model.type'] == model_tag)]['artifact_uri'][0]

        path_to_model = model_artifact + '/model'
        model = mlflow.pyfunc.load_model(model_artifact + '/model')

        return model

    def serve_model(self, model_tag, run_id=None, port=1234):
        #execute as bash command
        raise NotImplementedError 
        # to server a model be sure to check conda.yaml file - check all required packages
        #process = subprocess.Popen(f'mlflow models serve -m ./mlruns/0/{run_id}/artifacts/model -p 1234', stdout=subprocess.PIPE)
        #return process

    def get_preds(self, data, port=1234):
        #data should be result of call: pandas.DataFrame.to_json(..., orient='split')
        endpoint = f'http://127.0.0.1:{port}/invocations'
        headers = {'content-type': 'application/json; format=pandas-split'}
        responce = requests.post(endpoint, data=data, headers=headers)
        return responce.json()