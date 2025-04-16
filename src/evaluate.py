import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.
    RMSE is a measure of the average magnitude of the errors in a set of predictions.
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def calculate_r2_percentage(y_true, y_pred):
    """
    Calculate the R-squared (R²) score as a percentage.
    R² represents the proportion of the variance in the dependent variable
    that is predictable from the independent variables.
    """
    r2 = r2_score(y_true, y_pred)
    return r2 * 100  # Convert to percentage

def calculate_accuracy_with_tolerance(y_true, y_pred, tolerance=5):
    """
    Calculate the percentage of predictions that are within a specified tolerance of the true values.
    This is useful for assessing the practical accuracy of the model.
    """
    differences = np.abs(y_true - y_pred)
    within_tolerance = differences <= tolerance
    accuracy = np.mean(within_tolerance) * 100
    return accuracy

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of the model using RMSE, R² as percentage, and accuracy within a tolerance.
    """
    rmse = calculate_rmse(y_true, y_pred)
    r2_percentage = calculate_r2_percentage(y_true, y_pred)
    accuracy = calculate_accuracy_with_tolerance(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2_percentage:.2f}%")
    print(f"Accuracy within ±5: {accuracy:.2f}%")

# Example usage
if __name__ == "__main__":
    # Example true and predicted values
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 145, 195, 240, 310])

    evaluate_model(y_true, y_pred)
