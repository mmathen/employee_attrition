import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from employee_attrition_model.config.core import config
from employee_attrition_model.pipeline import  create_pipeline
from employee_attrition_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config_.training_data_file)
    # Define categorical features
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_columns.remove('attrition')
    categorical_features=categorical_columns
    
    print(f"Creating pipeline with categorical features: {categorical_features}")
    employee_attrition_pipe = create_pipeline(categorical_features=categorical_features)
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config_.features],     # predictors X
        data[config.model_config_.target],       # target y
        test_size = config.model_config_.test_size,
        random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
    )

    # Pipeline fitting
    employee_attrition_pipe.fit(X_train, y_train)
    y_pred = employee_attrition_pipe.predict(X_test)

    # Calculate the score/error
    train_accuracy = employee_attrition_pipe.score(X_train, y_train)
    test_accuracy = employee_attrition_pipe.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # persist trained model
    save_pipeline(pipeline_to_persist = employee_attrition_pipe)
    
if __name__ == "__main__":
    run_training()