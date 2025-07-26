# Model Card

## Model Details
- **Model type**: Random Forest Classifier
- **Version**: 1.0.0
- **Training date**: 2023
- **Parameters**:
  - n_estimators: 100
  - max_depth: 15
  - random_state: 42
  - n_jobs: -1 (use all available cores)
- **Features used**: 
  - Numerical: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

## Intended Use
- **Primary intended uses**: Predict whether a person's income exceeds $50K/year based on census data
- **Primary intended users**: Researchers, policy makers, and social scientists studying income inequality and socioeconomic factors
- **Out-of-scope uses**: This model should not be used for making decisions about individuals that could impact their livelihood, such as loan approvals, hiring decisions, or determining eligibility for benefits

## Training Data
- **Source**: Census Income dataset 
- **Collection process**: Extracted from the 1994 Census bureau database
- **Training data size**: 80% of the original dataset
- **Data preprocessing**: 
  - Categorical features were one-hot encoded
  - Labels were binarized (>50K and <=50K)
  - Missing values (represented as '?') were kept as a separate category

## Evaluation Data
- **Source**: Same as training data
- **Evaluation data size**: 20% of the original dataset
- **Data preprocessing**: Same preprocessing steps as training data
- **Evaluation procedure**: The model was evaluated on a held-out test set

## Metrics
- **Overall Performance (Full Dataset Evaluation)**:
  - Precision: 0.8458
  - Recall: 0.6249
  - F1 Score: 0.7188
- **Overall Performance (Test Dataset Evaluation)**:
  - Precision: 0.7918
  - Recall: 0.5786
  - F1 Score: 0.6686
- **Slice Performance**:
  - The model's performance varies across different demographic slices.
  - Detailed slice metrics for the full dataset are saved in `metrics/full_data_metrics.csv`.
  - Detailed slice metrics for the test dataset are saved in `metrics/test_data_metrics.csv`.
  - Notable observations:
    - Performance is better for majority demographic groups.
    - Lower performance for underrepresented groups in the training data.

## Ethical Considerations
- **Data bias**: The census data reflects historical biases in society, which may be perpetuated by the model
- **Fairness concerns**: 
  - The model may perform differently across different demographic groups
  - Performance disparities across race, gender, and other protected attributes should be carefully monitored
- **Mitigations**: 
  - Slice metrics are computed to identify and address performance disparities
  - Users should be aware of these limitations when interpreting model predictions

## Caveats and Recommendations
- **Limitations**:
  - The model is trained on US census data from 1994, which may not reflect current socioeconomic conditions
  - The binary income classification (>50K vs <=50K) is a simplification of the complex nature of income
  - The model does not account for regional cost of living differences
- **Recommendations**:
  - Regularly retrain the model with more recent data
  - Consider using more fine-grained income brackets for more nuanced predictions
  - Supplement model predictions with additional context and domain knowledge
  - Monitor performance across different demographic slices to ensure fairness