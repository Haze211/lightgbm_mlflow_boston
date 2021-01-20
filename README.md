# MLOps with Mlflow

This reposetory showcases an approach of handling machine learning project and simplifies process of tracking and logging different versions
of machine learning model and its hyperparameters. The main technology used for MLOps workflow is [Mlflow](https://www.mlflow.org/docs/latest/index.html). Mlflow takes care of logging model artifacts, exposing endpoint for model prediction and simplifies process of communication between data scientists that are working on the same project. [Lightgbm](https://lightgbm.readthedocs.io/en/latest/index.html) is the main model architecture used in the project. Dataset used in this project is **Boston Housing Dataset** available in [sklearn datasets.](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html). Main goal of this project is to utilize MLflow for better machine learning lifecycle. 

## Project structure
Repository consists of 4 main parts:
- MLproject : Specification file. Serves as a entry point for a project. Usining ``` mlflow run ``` creates new environment and installs packages specified in ``` conda.yaml ```  and executes command from ```main```. Its also possible to create several entry points, for example to validate or re-train model.
- conda.yaml : Specification file. Mainly responsible for package instalation. Mlflow can also utilize Docker env or default system env. [Mlflow Project Environments](https://www.mlflow.org/docs/latest/projects.html#project-environments)
- train.py : Main executable file. Consists of next steps:
    - Preparing training and validation data in suitable format for lightgbm.
    - Finding best model hyperparams using [hyperopt](https://github.com/hyperopt/hyperopt)
    - Training model with best set of hyperparams. Logging model artifacts into Mlflow.
- train_lightgbm_hyperopt.py : Support class which exposes methods for:
    - Preparing the data.
    - Defining objective and minimization function for hyperopt.
    - Finding best hyperparams.
    - Logging best model artifcats and tagging model for better search across possible big model pool.
    - Loading model based on tag or run_id for possible retraining or inference.
    - Getting predictions out of served model. 

## Example workflow

When working on a complex project its expected to have many iterations during training machine learning model. And its extremly easy to forgot what steps during preprocessing you have done, what hyperparameters you have used, especially if this params were handpicked based on business logic. And if you cannot ensure reproducability, you might easily forget what you have done. To simplify this whole rutine one must track preprocessing steps, evolution of the ML model, metrics during training and validaton, etc.  One way to do this is to ensure that you can easily reproduce steps that were taken. For dataset manipulation you can eiser log different tags with ```mlflow.set_tag``` which then will be visible in Mlflow run data or utilize something like [dvc](https://dvc.org/) which is data version control system, very similar to github.

For controling ML experiments one can utilize MLflow in next manner:
- Initialize new project with ```mlflow.create_experiment```
- Start mlflow run and log metrics, model params, tags using
```python
    with mlflow.start_run():
        #initialize model
        model = Model(...)
        #fit 
        model.fit(X, y)

        #get predictions
        preds = model.predict(X_test)

        #evaluate metrics
        metric = get_metric(true, preds)

        #Log results
        mlflow.log_param('param1', param1)
        mlflow.log_metric('metric', metric)
        
        #Log model and add it to model registry
        mlflow.sklearn.log_model(model, "model", registered_model_name="BestEverModel")
```
After finishing the ML training cycle, its possible to view results using Mlflow UI. Command should be executed from working directory. After calling ```mlflow.start_run``` Mlflow will create directory to store run data per experiment. If you dont specify name for new experiment, Mlflow will store everything in **Default** project.

```bash
mlflow ui
```
By default, Mlflow will use 5000 port, so opening http://localhost:5000/#/ will show the Mlflow UI.

Here is a [good overview](https://docs.databricks.com/applications/mlflow/tracking.html) of what you can do with the model in the UI. In short, you can compare different runs, save model, add it to model registry, set tags, change configs etc.

Once you model training is done, its easy to serve model, so it will act as an API and will accept calls and return predictions.
To serve a model you need to provide a **run_id** and optionaly a port. By default, Mlflow will open 1234 port for served model.

To serve model localy, execute 
```bash
mlflow models serve -m ./mlruns/0/run_id/artifacts/model -p 1234
```

After this point you can get predictions from served model using CURL request. Data param should be a result of calling 
```python
    pandas.DataFrame.to_json(..., orient='split')
```
on your dataset.

Get predictions from the model:
```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" 
--data 'data' 
http://127.0.0.1:1234/invocations
```

#### Usage
To run this project, invoke 

```bash
mlflow run lightgbm_mlflow_boston
```
Alternatively, you can execute project directly from github 

```bash
mlflow run https://github.com/Haze211/lightgbm_mlflow_boston.git
```

