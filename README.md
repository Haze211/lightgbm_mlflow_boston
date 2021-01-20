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

#### Usage
To run this project, invoke 

```bash
mlflow run lightgbm_mlflow_boston
```
Alternatively, you can execute project directly from github 

```bash
mlflow run https://github.com/Haze211/lightgbm_mlflow_boston.git
```

