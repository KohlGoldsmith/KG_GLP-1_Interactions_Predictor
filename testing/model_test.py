import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.symptom_helper import generate_symptom_vector, MAPPING_DF, EMBEDDING_DIM
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve

# Setting up style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(ROOT_DIR, "dataset", "processed", "faers_with_embeddings_ready.csv")

# 1. Load Data
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Dataset loaded: {df.shape}")

# 2. Dynamic Column Identification
reaction_cols = [c for c in df.columns if c.startswith('emb_')]
all_drug_cols = [c for c in df.columns if c.startswith('drug_')]
sex_cols = [c for c in df.columns if c.startswith('sex_')]
demo_cols = ['age'] + sex_cols

# 3. Handle GLP-1 Inputs
target_glp1_names = ['LIRAGLUTIDE', 'SEMAGLUTIDE', 'DULAGLUTIDE', 'EXENATIDE']
glp1_cols = []
for target in target_glp1_names:
    match = next((c for c in all_drug_cols if target.upper() in c.upper()), None)
    if match:
        glp1_cols.append(match)

target_drugs = [c for c in all_drug_cols if c not in glp1_cols]

# 4. Prepare X and y
X = df[reaction_cols + demo_cols + glp1_cols]
y = df[target_drugs]

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# 6. Train Neural Network model
# Scaling is crucial for Neural Networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Neural Network...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), #This defines the actual "neural" structure of the model.
                      activation='relu', # This is the mathematical function that decides if a neuron should "fire" or not.
                                         # If the input is negative, the output is 0 (the neuron is off).
                                         # If the input is positive, the output is the input (the neuron is on).
                      solver='adam', # (Adaptive Moment Estimation) combines the best properties of other optimizers.
                                     # It adapts the learning rate for each parameter individually.
                      max_iter=200, # This sets the maximum number of epochs the model is allowed to run.
                      random_state=22, # Arbitrary value
                      early_stopping=True)

model.fit(X_train_scaled, y_train)
print("Model trained.")

# 7. Optimization for thresholds
print("\nOptimizing thresholds for each drug...")
y_proba = model.predict_proba(X_test_scaled)  # Using the previously scaled data for prediction
best_thresholds = {}

# Check if model output is a List (RF) or Array (MLP)
is_list_output = isinstance(y_proba, list)

for i, col_name in enumerate(target_drugs):

    # Robust Logic for MLP and RF
    if is_list_output:
        # RF/MultiOutput: List of arrays
        prob_array = y_proba[i]
        if prob_array.shape[1] > 1:
            probs = prob_array[:, 1]
        else:
            probs = np.zeros(prob_array.shape[0])
    else:
        # MLP: Single Matrix (rows=patients, cols=drugs)
        probs = y_proba[:, i]

    # Threshold Calculation
    precision, recall, thresholds = precision_recall_curve(y_test[col_name], probs)

    numerator = 2 * precision * recall
    denominator = precision + recall
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    best_idx = np.nanargmax(f1_scores)

    if best_idx < len(thresholds):
        # This enforces a minimum threshold of 0.10 to prevent the model from spamming "Yes"
        # because your data is highly imbalanced.
        best_thresholds[col_name] = max(thresholds[best_idx], 0.10)
    else:
        best_thresholds[col_name] = 0.5

print("Optimization complete.")


# 8. Evaluate with thresholds
def custom_predict(model, X_input, thresholds, targets, is_list):
    probs_list = model.predict_proba(X_input)
    predictions = pd.DataFrame(index=pd.RangeIndex(len(X_input)), columns=targets)  # Safe index creation

    for i, col in enumerate(targets):
        if is_list:
            if probs_list[i].shape[1] > 1:
                p = probs_list[i][:, 1]
            else:
                p = np.zeros(len(X_input))
        else:
            p = probs_list[:, i]

        t = thresholds.get(col, 0.5)
        predictions[col] = (p >= t).astype(int)
    return predictions


# Pass scaled data to predict
y_pred_optimized = custom_predict(model, X_test_scaled, best_thresholds, target_drugs, is_list_output)

print("\n=== Classification Report (Optimized) ===")
print(classification_report(y_test, y_pred_optimized, target_names=target_drugs, zero_division=0))

# User Input and Predictions

print("\nGenerating Prediction for User Profile:")
# Manually change these parameters totest for different side effects. This would be placed as a dropdown if on a webapp
user_symptoms = ["Abdominal pain", "Vomiting", "Dizziness"]

# A. Generate Symptom Embeddings
mean_embedding = generate_symptom_vector(user_symptoms, MAPPING_DF)
emb_features = {f"emb_{i}": mean_embedding[i] for i in range(EMBEDDING_DIM)}

# B. Construct User Input
# Change these parameters manually to test different drugs and age or sex of subject.
user_input_base = {'age': 55}
for col in sex_cols: user_input_base[col] = 0
male_col = next((c for c in sex_cols if 'male' in c.lower() and 'female' not in c.lower()), None)
if male_col: user_input_base[male_col] = 1

for col in glp1_cols:
    user_input_base[col] = 0
    if 'SEMAGLUTIDE' in col.upper(): user_input_base[col] = 1

# C. Combine
final_user_input = {**user_input_base, **emb_features}
user_df = pd.DataFrame([final_user_input], columns=X.columns).fillna(0)

# User input must be scaled just like training data
user_df_scaled = scaler.transform(user_df)

# D. Get Probabilities (Robust MLP Support)
probs_list = model.predict_proba(user_df_scaled)
is_list_output = isinstance(probs_list, list)

drug_probs = []
for i, drug_col in enumerate(target_drugs):
    if is_list_output:
        # RF Style
        if probs_list[i].shape[1] > 1:
            score = probs_list[i][:, 1][0]
        else:
            score = 0.0
    else:
        # MLP Style (1 row, N columns)
        score = probs_list[0, i]

    drug_probs.append(score)

# Results
results = pd.Series(drug_probs, index=target_drugs)
top_results = results.sort_values(ascending=False).head(20)

print("\nTop 10 Predicted Co-Occurring Drugs (Raw Probabilities):")
print(top_results.head(10))

# Visualizations
plt.figure(figsize=(10, 8))
sns.barplot(x=top_results.values, y=top_results.index, palette="viridis")
plt.title(f"Top 20 Predicted Drug Associations (Neural Net)\n(GLP-1: Semaglutide, Age: 55, User Symptoms: ", user_symptoms, ")")
plt.xlabel("Predicted Probability")
plt.ylabel("Drug Name")
plt.tight_layout()
plt.show()