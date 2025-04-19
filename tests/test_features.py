
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
import pytest
from employee_attrition_model.config.core import config
from employee_attrition_model.processing.features import LabelEncoderTransformer, FeatureSelectorLabelled, FeatureSelector_ANOVA, CorrelationReducer


def create_dummy_dataframe():
    data = {
        'numerical_feature_1': [10, 20, 30, 40, 50, 60],
        'numerical_feature_2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'categorical_feature_1': ['A', 'B', 'A', 'C', 'B', 'A'],
        'categorical_feature_2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
        'constant_feature': [100, 100, 100, 100, 100, 100],
        'high_corr_feature_1': [1, 2, 3, 4, 5, 6],
        'high_corr_feature_2': [2, 4, 6, 8, 10, 12], # Highly correlated with high_corr_feature_1
        'target': [0, 1, 0, 1, 0, 1] # Dummy target for classifiers
    }
    df = pd.DataFrame(data)
    # Ensure some columns are 'object' and some could be 'category' if needed
    df['categorical_feature_1'] = df['categorical_feature_1'].astype('object')
    df['categorical_feature_2'] = df['categorical_feature_2'].astype('category')
    return df

# ==============================================================================
# LabelEncoderTransformer Tests
# ==============================================================================

def test_label_encoder_fit():
    df = create_dummy_dataframe()
    # Pass all columns to the transformer, it should identify categoricals
    transformer = LabelEncoderTransformer(columns=df.columns.tolist())
    transformer.fit(df)

    assert set(transformer.cat_columns) == set(['categorical_feature_1', 'categorical_feature_2'])
    assert 'categorical_feature_1' in transformer.encoders
    assert 'categorical_feature_2' in transformer.encoders
    assert len(transformer.encoders['categorical_feature_1'].classes_) == 3 # A, B, C
    assert len(transformer.encoders['categorical_feature_2'].classes_) == 2 # X, Y

# def test_label_encoder_transform():
#     df = create_dummy_dataframe()
#     transformer = LabelEncoderTransformer(columns=df.columns.tolist())
#     transformer.fit(df)
#     df_transformed = transformer.transform(df)

    # assert isinstance(df_transformed, pd.DataFrame)
    # assert list(df_transformed.columns) == list(df.columns) # Ensure all original columns are present

    # # Check that categorical columns are encoded
    # assert df_transformed['categorical_feature_1'].dtype == 'int64' # Check dtype after fillna(-1).astype(int)
    # assert df_transformed['categorical_feature_2'].dtype == 'int64'

    # # Check encoded values (based on sorted unique values: A=0, B=1, C=2; X=0, Y=1)
    # assert df_transformed['categorical_feature_1'].tolist() == [0, 1, 0, 2, 1, 0]
    # assert df_transformed['categorical_feature_2'].tolist() == [0, 1, 0, 1, 0, 1]

    # # Check that numerical and constant columns are unchanged
    # pd.testing.assert_series_equal(df_transformed['numerical_feature_1'], df['numerical_feature_1'])
    # pd.testing.assert_series_equal(df_transformed['constant_feature'], df['constant_feature'])

def test_label_encoder_unseen_categories():
    df = create_dummy_dataframe()
    transformer = LabelEncoderTransformer(columns=df.columns.tolist())
    transformer.fit(df)

    # Create new data with an unseen category
    new_data = {'categorical_feature_1': ['A', 'D'], 'categorical_feature_2': ['X', 'Z'], 'numerical_feature_1': [1, 2]}
    new_df = pd.DataFrame(new_data)

    
    full_new_data = {col: new_df.get(col, df[col].iloc[:len(new_df)].tolist()) for col in df.columns}
    full_new_df = pd.DataFrame(full_new_data)


    df_transformed = transformer.transform(full_new_df)

    # Check how unseen categories are handled (should be -1 due to fillna(-1))
    # categorical_feature_1: A=0, D=unseen -> -1
    # categorical_feature_2: X=0, Z=unseen -> -1
    assert df_transformed['categorical_feature_1'].tolist()[:2] == [0, -1]
    assert df_transformed['categorical_feature_2'].tolist()[:2] == [0, -1]


# ==============================================================================
# FeatureSelectorLabelled Tests
# ==============================================================================

def test_feature_selector_labelled_fit():
    df = create_dummy_dataframe()
    transformer = FeatureSelectorLabelled()
    transformer.fit(df)

    assert set(transformer.columns_to_drop_) == set(['constant_feature'])

def test_feature_selector_labelled_transform():
    df = create_dummy_dataframe()
    transformer = FeatureSelectorLabelled()
    transformer.fit(df)
    df_transformed = transformer.transform(df)

    assert 'constant_feature' not in df_transformed.columns
    assert set(df_transformed.columns) == set([col for col in df.columns if col != 'constant_feature'])
    assert len(df_transformed.columns) == len(df.columns) - 1

def test_feature_selector_labelled_no_constant_features():
    df = create_dummy_dataframe().drop(columns=['constant_feature'])
    transformer = FeatureSelectorLabelled()
    transformer.fit(df)
    df_transformed = transformer.transform(df)

    assert len(transformer.columns_to_drop_) == 0
    pd.testing.assert_frame_equal(df_transformed, df) # No columns should be dropped

# ==============================================================================
# FeatureSelector_ANOVA Tests
# ==============================================================================



# def test_feature_selector_anova_fit_transform_k_less_than_n_features():
#     # Create numerical data suitable for ANOVA
#     df_numerical = pd.DataFrame({
#         'num_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#         'num_2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
#         'num_3': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5], # Low variance
#         'num_4': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], # Moderate variance
#         'num_5': [1, 5, 2, 6, 3, 7, 4, 8, 5, 9], # Higher variance
#     })
#     # Categorical target
#     y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#     k = 3 # Select top 3 features

#     transformer = FeatureSelector_ANOVA(k=k)
#         # When
#     # Explicitly capture the UserWarning and RuntimeWarning triggered by the constant feature
#     with pytest.warns(UserWarning) as user_warnings:
#          with pytest.warns(RuntimeWarning) as runtime_warnings:
#               transformer.fit(df_numerical, y)

#     # Then
#     # Assert that the expected warnings occurred
#     assert len(user_warnings) == 1 # Expecting one UserWarning about constant features
#     assert "Features [2] are constant." in str(user_warnings[0].message)

#     assert len(runtime_warnings) == 1 # Expecting one RuntimeWarning for invalid value
#     assert "invalid value encountered in divide" in str(runtime_warnings[0].message)
    
#     transformer.fit(df_numerical, y)
#     df_transformed = transformer.transform(df_numerical)

#     assert len(transformer.selected_features_) == k
#     assert len(df_transformed.columns) == k
#     assert set(df_transformed.columns).issubset(set(df_numerical.columns))

def test_feature_selector_anova_k_greater_than_n_features():
    # Create numerical data with fewer features than k
    df_numerical = pd.DataFrame({
        'num_1': [1, 2, 3, 4, 5],
        'num_2': [5, 4, 3, 2, 1],
    })
    y = pd.Series([0, 1, 0, 1, 0])

    k = 5 # k is greater than the number of features (2)

    transformer = FeatureSelector_ANOVA(k=k)

    # Expect a UserWarning
    with pytest.warns(UserWarning, match=f"k={k} is greater than n_features={df_numerical.shape[1]}. All the features will be returned."):
        transformer.fit(df_numerical, y)

    df_transformed = transformer.transform(df_numerical)

    assert len(transformer.selected_features_) == df_numerical.shape[1] # Should return all features
    assert len(df_transformed.columns) == df_numerical.shape[1]
    pd.testing.assert_frame_equal(df_transformed, df_numerical) # Should return the original DataFrame

def test_feature_selector_anova_fit_sets_attribute():
    df_numerical = pd.DataFrame({'num_1': [1, 2, 3], 'num_2': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    transformer = FeatureSelector_ANOVA(k=1)
    transformer.fit(df_numerical, y)
    assert isinstance(transformer.selected_features_, pd.Index)
    assert len(transformer.selected_features_) == 1

# ==============================================================================
# CorrelationReducer Tests
# ==============================================================================

# def test_correlation_reducer_fit_transform():
#     df = create_dummy_dataframe().drop(columns=['constant_feature', 'categorical_feature_1', 'categorical_feature_2', 'target'])
#     # numerical_feature_1 vs high_corr_feature_1 vs high_corr_feature_2 should be highly correlated
#     # numerical_feature_2 is not highly correlated with others in this simple example
#     threshold = 0.75
#     reducer = CorrelationReducer(threshold=threshold)
#     reducer.fit(df)
#     df_transformed = reducer.transform(df)

    # Check that at least one of the highly correlated features is dropped.
    
    # remaining_cols = df_transformed.columns.tolist()

   
    # corr_matrix = df.corr()
    # abs(corr_matrix.loc['high_corr_feature_1', 'high_corr_feature_2']) is 1.0
    # abs(corr_matrix.loc['numerical_feature_1', 'high_corr_feature_1']) is 1.0
    # abs(corr_matrix.loc['numerical_feature_1', 'high_corr_feature_2']) is 1.0

    # The logic in fit will find correlations:
    # (numerical_feature_1, high_corr_feature_1) -> add high_corr_feature_1 to set
    # (numerical_feature_1, high_corr_feature_2) -> add high_corr_feature_2 to set
    # (high_corr_feature_1, high_corr_feature_2) -> add high_corr_feature_2 to set

    # So, correlated_features should contain {'high_corr_feature_1', 'high_corr_feature_2'}
    # assert set(reducer.correlated_features) == set(['high_corr_feature_1', 'high_corr_feature_2'])

    # # The transformed dataframe should not contain these
    # assert 'high_corr_feature_1' not in df_transformed.columns
    # assert 'high_corr_feature_2' not in df_transformed.columns

    # The remaining columns should be the others
    #assert set(df_transformed.columns) == set(['numerical_feature_1', 'numerical_feature_2'])

def test_correlation_reducer_no_highly_correlated_features():
    df = pd.DataFrame({
        'num_1': [1, 2, 3, 4, 5],
        'num_2': [5, 4, 3, 2, 1], # Perfect negative correlation (abs=1)
        'num_3': [1, 1, 1, 2, 2], # Low correlation
    })
    threshold = 0.9 # Set a high threshold

    reducer = CorrelationReducer(threshold=threshold)
    reducer.fit(df)
    df_transformed = reducer.transform(df)

    # Although num_1 and num_2 have perfect negative correlation,
    # the threshold is 0.9, so their absolute correlation is >= threshold.
    # The loop structure should identify (num_1, num_2) -> add num_2 to set.
    assert set(reducer.correlated_features) == set(['num_2'])
    assert 'num_2' not in df_transformed.columns
    assert set(df_transformed.columns) == set(['num_1', 'num_3'])

    # Test with a threshold > 1 (no features should be dropped)
    reducer_high_thresh = CorrelationReducer(threshold=1.1)
    reducer_high_thresh.fit(df)
    df_transformed_high_thresh = reducer_high_thresh.transform(df)
    assert len(reducer_high_thresh.correlated_features) == 0
    pd.testing.assert_frame_equal(df_transformed_high_thresh, df)


def test_correlation_reducer_fit_sets_attribute():
    df = create_dummy_dataframe().drop(columns=['constant_feature', 'categorical_feature_1', 'categorical_feature_2', 'target'])
    reducer = CorrelationReducer(threshold=0.75)
    reducer.fit(df)
    assert isinstance(reducer.correlated_features, set)
    # Content check is done in the transform test

