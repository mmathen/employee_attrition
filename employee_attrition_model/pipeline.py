import sys
from pathlib import Path
import pandas as pd
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
#from sklearn.ensemble import CatBoostClassifier
from catboost import CatBoostClassifier

from employee_attrition_model.config.core import config
from employee_attrition_model.processing.features import LabelEncoderTransformer, FeatureSelectorLabelled, FeatureSelector_ANOVA,CorrelationReducer

def _convert_to_dataframe_pre_ct(X):
    """Converts input X to a Pandas DataFrame before ColumnTransformer."""
  
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X) # This creates a DataFrame, but column names might be lost if X was array


def _convert_to_dataframe_post_ct(X):
    """Converts input X to a Pandas DataFrame after ColumnTransformer."""
    #
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)

def create_pipeline(categorical_features: list) -> Pipeline:
    """
    Creates and returns the employee attrition model pipeline.

    Args:
        categorical_features: A list of names of the categorical features.
    """

    # Ensure 'attrition' is not in the list as it's the target
    pipeline_categorical_features = [col for col in categorical_features if col != 'attrition']


    employee_attrition_pipe = Pipeline([
        #('to_dataframe_before_label_encoding', FunctionTransformer(lambda X: pd.DataFrame(X, columns=pipeline_categorical_features ))), # Use pipeline_categorical_features here
        ('label_encoding', ColumnTransformer(
            transformers=[
                ('label_encoder', LabelEncoderTransformer(columns=pipeline_categorical_features), pipeline_categorical_features) # Use pipeline_categorical_features here
            ],
            remainder='passthrough'
        )),

        
        ('to_dataframe', FunctionTransformer(func=_convert_to_dataframe_post_ct)), # Convert to DataFrame after ColumnTransformer, columns might be different now
        # You might need to re-assign column names if FeatureSelectorLabelled depends on them

        ('feature_selection_labelled', FeatureSelectorLabelled()),
        ('feature_selection_anova', FeatureSelector_ANOVA(k=config.model_config_.anova_k)), 
        ('correlation_reducer', CorrelationReducer(threshold=config.model_config_.correlation_threshold)), # Assuming threshold comes from config

        ('catboost', CatBoostClassifier(
            custom_loss=[metrics.Accuracy()],
            random_seed=config.model_config_.random_state, # Assuming random_state comes from config
            logging_level='Silent',
            
        )),
    ])

    return employee_attrition_pipe

    employee_attrition_pipe_default = create_pipeline(config.model_config_.default_categorical_features) # Example if you have default features in config
# employee_attrition_pipe = Pipeline([
#     ('to_dataframe_before_label_encoding', FunctionTransformer(lambda X: pd.DataFrame(X, columns=[col for col in categorical_features if col != 'attrition'] ))),
#     ('label_encoding', ColumnTransformer(
#         transformers=[
#             ('label_encoder', LabelEncoderTransformer(columns=categorical_features), categorical_features)
#         ],
#         remainder='passthrough'  
#     )),
  
#     ('to_dataframe', FunctionTransformer(lambda X: pd.DataFrame(X, columns=categorical_features))), # convert to DataFrame
#     ('feature_selection_labelled', FeatureSelectorLabelled()),  # Apply FeatureSelectorLabelled here
#     ('feature_selection_anova', FeatureSelector_ANOVA(k=15)),  # Apply FeatureSelector_ANOVA here
#     ('correlation_reducer', CorrelationReducer(threshold=0.75)),
#     # Add your CatBoost or XGBoost model as the final step
#     ('catboost', CatBoostClassifier(custom_loss=[metrics.Accuracy()], random_seed=42, logging_level='Silent')),
#     # ('xgboost', XGBClassifier(n_estimators=10, max_depth=8, learning_rate=0.2, objective='binary:logistic'))
# ])
