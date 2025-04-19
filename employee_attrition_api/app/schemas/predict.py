from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[ List[str]]


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
