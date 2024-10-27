import pandas as pd
from pycaret.classification import *

# Load CSV File
data = pd.read_csv('Preprocessed_data1b.csv')
# Preview the dataset
print(data.head())

# Initialize setup
clf = setup(data=data, 
            target='Student_final_result',  # Actual target column
            train_size=0.7,  # 70% training, 30% testing
            normalize=True,  # normalize the features
            session_id=42)  # session_id is for reproducibility

best_models = compare_models(n_select=3)

# Open a text file to save model names
with open('top_models.txt', 'w') as file:
    for i, model in enumerate(best_models):
        model_name = str(model).split('(')[0]  # Extract model name from string representation
        file.write(f"Model {i+1}: {model_name}\n")  # Write model name to the file
        print(f"Model {i+1}: {model_name}")  # Print to console for confirmation

# List to hold tuned models
tuned_models = []
# Loop through each selected model, tune it, and save it
for i, model in enumerate(best_models):
    tuned_model = tune_model(model)
    tuned_models.append(tuned_model)
    # Save the tuned model with its name
    model_name = str(tuned_model).split('(')[0]  # Extract the model name
    save_model(tuned_model, model_name)

for model in tuned_models:
    evaluate_model(model)