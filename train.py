from train_lightgbm_hyperopt import *

import os
import warnings
import sys




if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(40)
    #unpack default args
    maxevals = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    project_name = str(sys.argv[2]) if len(sys.argv) > 2 else "Boston Housing"
    model_tag = str(sys.argv[2]) if len(sys.argv) > 3 else "production.1"
    
    print('========== Initializing training ========')
    main_class = LGBHyperoptProd()

    #get datasets
    train_data, validation_data, scaler = main_class.prepare_lgb_dataset()

    #get hyperparams search space
    search_space = main_class.get_optim_space()

    #get objective funtion
    obj_funct = main_class.get_optim_objective(train_data)

    print('========= Searching for best hyperparameters =========')
    best_hyperparams, trials = main_class. \
        get_optimal_model_hyperparams(obj_funct,
        search_space,
        train_data, 
        maxevals = maxevals)
    
    print(f"Best hyperparams: {best_hyperparams}")

    print("========= Training model with best hyperparams =========")
    main_class. \
        tag_model_for_production(project_name,
         train_data, 
         best_hyperparams, 
         trials, model_tag)