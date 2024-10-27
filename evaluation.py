from pycaret.classification import *
import pandas as pd

# # Load model and make predictions (assuming model and dataset are available)
data = pd.read_csv('Preprocessed_data1b.csv')

# # Load the final model
final_model = load_model('GradientBoostingClassifier1')

# # Make predictions
predictions = predict_model(final_model, data)

# Display predictions
print(predictions)