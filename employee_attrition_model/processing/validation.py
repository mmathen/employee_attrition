import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union,Any

from datetime import datetime
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from employee_attrition_model.config.core import config
from employee_attrition_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame = input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    age: Optional[Any] 
    businesstravel: Optional[Any] 
    dailyrate: Optional[Any] 
    department: Optional[Any] 
    distancefromhome: Optional[Any] 
    education: Optional[Any] 
    educationfield: Optional[Any] 
    employeecount: Optional[Any] 
   
    employeenumber: Optional[Any]
    environmentsatisfaction: Optional[Any] 
    gender: Optional[Any] 
    hourlyrate: Optional[Any] 
    jobinvolvement: Optional[Any] 
    joblevel: Optional[Any] 
    jobrole: Optional[Any] 
    jobsatisfaction: Optional[Any] 
    maritalstatus: Optional[Any] 
    monthlyincome: Optional[Any] 
    monthlyrate: Optional[Any] 
    numcompaniesworked: Optional[Any] 
    over18: Optional[Any] 
    overtime: Optional[Any] 
    percentsalaryhike: Optional[Any] 
    performancerating: Optional[Any]
    relationshipsatisfaction: Optional[Any] 
    standardhours: Optional[Any] 
    stockoptionlevel: Optional[Any] 
    totalworkingyears: Optional[Any]
    trainingtimeslastyear: Optional[Any] 
    worklifebalance: Optional[Any]
    yearsatcompany: Optional[Any] 
    yearsincurrentrole: Optional[Any] 
    yearssincelastpromotion: Optional[Any] 
    yearswithcurrmanager: Optional[Any] 
    
    


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]