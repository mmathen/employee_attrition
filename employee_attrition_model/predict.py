import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from employee_attrition_model import __version__ as _version
from employee_attrition_model.config.core import config
from employee_attrition_model.processing.data_manager import load_pipeline
from employee_attrition_model.processing.data_manager import pre_pipeline_preparation
from employee_attrition_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
employee_attrition_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    input_df=pd.DataFrame(input_data)
    input_df.columns = input_df.columns.str.lower()
    print("\n--- Columns of DataFrame passed to validate_inputs ---")
    print(input_df.columns)
    print("----------------------------------------------------")
    validated_data, errors = validate_inputs(input_df=input_df )
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = employee_attrition_pipe.predict(validated_data)
        results = {"predictions": predictions, "version": _version, "errors": errors}
        print("Results: " , results)
    print("Results String Value: " , results["predictions"][0])
    return results



if __name__ == "__main__":

    # data_in = {'dteday': ['2012-11-6'], 'season': ['winter'], 'hr': ['6pm'], 'holiday': ['No'], 'weekday': ['Tue'],
    #            'workingday': ['Yes'], 'weathersit': ['Clear'], 'temp': [16], 'atemp': [17.5], 'hum': [30], 'windspeed': [10]}
    data_in =pd.DataFrame({
    'Age': [35],
    'BusinessTravel': ['Travel_Rarely'],
    'DailyRate': [1102],
    'Department': ['Sales'],
    'DistanceFromHome': [1],
    'Education': [4], 
    'EducationField': ['Life Sciences'],
    'EmployeeCount': [1],
    'EmployeeNumber': [1001],
    'EnvironmentSatisfaction': [3],
    'Gender': ['Male'],
    'HourlyRate': [67],
    'JobInvolvement': [3],
    'JobLevel': [2],
    'JobRole': ['Sales Executive'],
    'JobSatisfaction': [4],
    'MaritalStatus': ['Married'],
    'MonthlyIncome': [5993],
    'MonthlyRate': [19479],
    'NumCompaniesWorked': [8],
    'Over18': ['Y'],
    'OverTime': ['Yes'],
    'PercentSalaryHike': [11],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [1],
    'StandardHours': [80],
    'StockOptionLevel': [0],
    'TotalWorkingYears': [8],
    'TrainingTimesLastYear': [0],
    'WorkLifeBalance': [1],
    'YearsAtCompany': [6],
    'YearsInCurrentRole': [4],
    'YearsSinceLastPromotion': [0],
    'YearsWithCurrManager': [5]
    }, index=[0])
    make_prediction(input_data = data_in)