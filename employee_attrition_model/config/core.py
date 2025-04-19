# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import employee_attrition_model

# Project Directories
PACKAGE_ROOT = Path(employee_attrition_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    pipeline_name: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    #unused_fields: List[str]
    
    

  
    # age_var: int
    # attrition_var: object
    # businesstravel_var: object
    # dailyrate_var: int
    # department_var: object
    # distancefromhome_var: int
    # education_var: int
    # educationfield_var: object
    # employeecount_var: int
    # employeenumber_var: int
    # environmentsatisfaction_var: int
    # gender_var: object
    # hourlyrate_var: int
    # jobinvolvement_var: int
    # joblevel_var: int
    # jobrole_var: object
    # jobsatisfaction_var: int
    # maritalstatus_var: object
    # monthlyincome_var: int
    # monthlyrate_var: int
    # numcompaniesworked_var: int
    # over18_var: object
    # overtime_var: object
    # percentsalaryhike_var: int
    # performancerating_var: int
    # relationshipsatisfaction_var: int
    # standardhours_var: int
    # stockoptionlevel_var: int
    # totalworkingyears_var: int
    # trainingtimeslastyear_var: int
    # worklifebalance_var: int
    # yearsatcompany_var: int
    # yearsincurrentrole_var: int
    # yearssincelastpromotion_var: int
    # yearswithcurrmanager_var: int

    
    test_size:float
    random_state: int
    random_seed: int
    logging_level: str
    model_type: str
    anova_k: int
    correlation_threshold: float
    #default_categorical_features: List[str] = []  # Default value for categorical features


class Config(BaseModel):
    """Master config object."""

    app_config_: AppConfig
    model_config_: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config_ = AppConfig(**parsed_config.data),
        model_config_ = ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()