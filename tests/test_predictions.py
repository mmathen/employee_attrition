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
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score

from employee_attrition_model.predict import make_prediction


def test_make_prediction_classification(sample_input_data):
    # Given
    # sample_input_data[0] is the input DataFrame (features)
    # sample_input_data[1] is the true target Series (original string labels like 'Yes'/'No')
    input_dataframe = sample_input_data[0]
    y_true_string = sample_input_data[1]

    # Expected number of predictions should match the number of input rows
    expected_num_of_predictions = len(input_dataframe)

    # Define an acceptable threshold for accuracy
    # This threshold depends on your model's performance; adjust as needed.
    # A value > 0.5 indicates better than random guessing for binary classification.
    acceptable_accuracy_threshold = 0.75 # Example threshold

    # When
    # Call the make_prediction function with the input features DataFrame
    result = make_prediction(input_data=input_dataframe)

    # Then
    # 1. Assert the overall result structure is a dictionary
    assert isinstance(result, dict)

    # 2. Assert expected keys are present in the result dictionary
    assert "predictions" in result # Should contain string predictions
    # REMOVE OR COMMENT OUT THIS LINE: assert "predictions_numerical" in result # This key is not returned

    assert "version" in result
    assert "errors" in result

    # 3. Assert that there were no validation errors in the input data
    assert result.get("errors") is None # Expecting successful validation for this test case

    # 4. Get the string predictions
    string_predictions = result.get("predictions")
    # Check if predictions is a NumPy array (based on your output)
    assert isinstance(string_predictions, np.ndarray)
    assert len(string_predictions) == expected_num_of_predictions # Check number of predictions

    # Check the type of individual predictions (should be strings)
    if string_predictions.size > 0: # Check if the array is not empty
         # Check dtype is object for strings
         assert string_predictions.dtype == 'object'
         # Optionally, check that prediction values are among the expected labels ('Yes', 'No', etc.)
         # expected_labels = ['Yes', 'No'] # Example based on your target
         # assert all(pred in expected_labels for pred in string_predictions)


    # 5. Calculate Classification Accuracy
    # Use the true string labels (y_true_string) and the predicted string labels (string_predictions - NumPy array)
    # accuracy_score can compare Series/arrays of strings directly if they match.
    try:
        # Ensure y_true_string is in a format compatible with accuracy_score and string_predictions (NumPy array)
        # Converting y_true_string Series to NumPy array for direct comparison is safe
        y_true_string_array = y_true_string.values if isinstance(y_true_string, pd.Series) else np.asarray(y_true_string)

        # Calculate accuracy comparing string labels (NumPy arrays)
        accuracy = accuracy_score(y_true_string_array, string_predictions)

        print(f"\nCalculated Classification Accuracy: {accuracy:.4f}") # Print accuracy for debugging


        # 6. Assert that the calculated accuracy is above the acceptable threshold
        assert accuracy >= acceptable_accuracy_threshold

    except Exception as e:
        pytest.fail(f"Error calculating or asserting accuracy: {e}")