# MLOps with Mlflow

This reposetory showcases an approach of handling machine learning project and simplifies process of tracking and logging different versions
of machine learning model and its hyperparameters. The main technology used for MLOps workflow is [Mlflow](https://www.mlflow.org/docs/latest/index.html). Mlflow takes care of logging model artifacts, exposing endpoint for model prediction and simplifies process of communication between data scientists that are working on the same project.

## Project structure
Repository consists of 4 main parts:
- MLproject : Specification file. Serves as a entry point for a project. Usining ``` mlflow run ``` creates new environment and installs packages specified in ``` conda.yaml ``` 
- conda.yaml
- train.py
- train_lightgbm_hyperopt.py


#### Usage
To run this project, invoke 

```bash
mlflow run lightgbm_mlflow_boston
```
Alternatively, you can execute project directly from github 

```bash
mlflow run https://github.com/Haze211/lightgbm_mlflow_boston.git
```

