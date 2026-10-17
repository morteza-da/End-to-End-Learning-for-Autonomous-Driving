# Threshold-Based Accuracy (research implementation)

**Source:** Supplied PhD research, PDF page 110, Equation (3-6).

For each continuous regression output, calculate the percentage of predictions whose **absolute error is within an inclusive threshold**:

- Steering: |predicted steering - reference steering| <= **6 degrees**
- Speed: |predicted speed - reference speed| <= **5 km/h**

`accuracy_percent = 100 * acceptable_prediction_count / total_prediction_count`

These are **two separate metrics**; the research passage does not define a combined joint success rate. The thresholds must be applied to predictions in the physical units shown above. Predictions stored in normalized units must first be inverse-transformed.

## Python example

```python
from autonomous_driving.threshold_metrics import driving_threshold_accuracies

result = driving_threshold_accuracies(
    steering_true_deg=[0, 10, -10],
    steering_pred_deg=[5, 17, -10],
    speed_true_kmh=[20, 30, 40],
    speed_pred_kmh=[25, 36, 38],
)
print(result)
# {'steering_accuracy_percent': 66.66666666666667,
#  'speed_accuracy_percent': 66.66666666666667}
```

**Important:** The example is synthetic, not a previous research measurement. Threshold accuracy is not RMSE, R², or classification accuracy. Results from new experiments must not be represented as reproductions of research-reported values without actual validation.

Run tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```
