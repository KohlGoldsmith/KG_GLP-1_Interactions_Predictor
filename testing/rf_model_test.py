# This is a Random Forest model to analyze the drugs taken and user profile.
# Manual input is being used for now but could be better implemented into a webapp.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.symptom_helper import generate_symptom_vector
from utils.symptom_helper import MAPPING_DF
from utils.symptom_helper import EMBEDDING_DIM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

# Setting up style for plots using seaborn
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

# Identify all drug columns present in the CSV
all_drug_cols = [c for c in df.columns if c.startswith('drug_')]

# Identify sex columns dynamically (handles sex_Male, sex_1, etc.)
sex_cols = [c for c in df.columns if c.startswith('sex_')]
demo_cols = ['age'] + sex_cols

print(f"Identified {len(sex_cols)} sex columns: {sex_cols}")

# 3. Handle GLP-1 Inputs

target_glp1_names = ['LIRAGLUTIDE', 'SEMAGLUTIDE', 'DULAGLUTIDE', 'EXENATIDE']

glp1_cols = []
for target in target_glp1_names:
    # Check for drug_LIRAGLUTIDE, drug_Liraglutide, etc.
    match = next((c for c in all_drug_cols if target.upper() in c.upper()), None)
    if match:
        glp1_cols.append(match)

print(f"GLP-1 Drugs found in dataset: {glp1_cols}")

# Define Targets: All drugs EXCEPT the GLP-1 drugs
target_drugs = [c for c in all_drug_cols if c not in glp1_cols]

# 4. Prepare X and y
X = df[reaction_cols + demo_cols + glp1_cols]
y = df[target_drugs]

print(f"Inputs (X): {X.shape[1]} features")
print(f"Outputs (y): {y.shape[1]} drugs to predict")

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# 6. Train Model

forest = RandomForestClassifier(n_estimators=30, random_state=22, class_weight='balanced', n_jobs=-1)
model = MultiOutputClassifier(forest)

print("Training model...")
model.fit(X_train, y_train)
print("Model trained.")

# 7. Evaluate

y_pred = model.predict(X_test)
print("\n=== Classification Report (Weighted Average Only) ===")
print(classification_report(y_test, y_pred, target_names=target_drugs, zero_division=0))

## ------------------------------------------------------------------------------------
# User Input and Prediction
# This is where the manual input must take place
## ------------------------------------------------------------------------------------

print("\nGenerating Prediction for User Profile")

#Input symptoms here, for testing purposes you must know about symptoms from the standard list used.

user_symptoms = ["Abdominal pain", "Vomiting", "Dizziness"]

# A. Generate Symptom Embeddings

mean_embedding = generate_symptom_vector(user_symptoms, MAPPING_DF)
emb_features = {f"emb_{i}": mean_embedding[i] for i in range(EMBEDDING_DIM)}

# B. Construct User Input

user_input_base = {'age': 55}

# Set Sex: Set all sex columns to 0, then set the specific Male column to 1 if found

for col in sex_cols:
    user_input_base[col] = 0
# specific logic: find the column for Male
male_col = next((c for c in sex_cols if 'male' in c.lower() and 'female' not in c.lower()), None)
if male_col:
    user_input_base[male_col] = 1 # Set Male to 1

# Set Drugs: Set all GLP-1 cols to 0, set target drug (in this case Semaglutide) to 1

for col in glp1_cols:
    user_input_base[col] = 0
    # Check if this column is semaglutide
    if 'SEMAGLUTIDE' in col.upper():
        user_input_base[col] = 1

# C. Combine and DataFrame

final_user_input = {**user_input_base, **emb_features}

# Create DataFrame using X.columns to ensure exact alignment of features

user_df = pd.DataFrame([final_user_input], columns=X.columns).fillna(0)
print(f"Prediction input shape: {user_df.shape}")

# D. Get Probabilities
probs_list = model.predict_proba(user_df)

# Extract probability of class "1" for each drug

drug_probs = []
for i, drug_col in enumerate(target_drugs):
    prob_array = probs_list[i]
    # If the model knows about both classes (0 and 1), take the probability of 1
    if prob_array.shape[1] > 1:
        score = prob_array[:, 1][0]
    else:
        # If model only sees class 0, prob is 0. If only class 1, prob is 1.
        # usually prob_array is [[prob_class_0]]
        score = 0.0
    drug_probs.append(score)

results = pd.Series(drug_probs, index=target_drugs)
top_results = results.sort_values(ascending=False).head(20)

print("\nTop 10 Predicted Co-Occurring Drugs (Interaction Candidates):")
print(top_results.head(10))

# Visualizations

# 1. Bar Chart
plt.figure(figsize=(10, 8))
sns.barplot(x=top_results.values, y=top_results.index, palette="viridis")
plt.title(f"Top 20 Predicted Drug Associations\n(GLP-1: Semaglutide, Age: 55, Symptoms: {', '.join(user_symptoms)})")
plt.xlabel("Predicted Probability")
plt.ylabel("Drug Name")
plt.tight_layout()
plt.show()

# 2. Risk Heatmap
try:
    # Ensure exactly 20 items for the reshape
    if len(top_results) >= 20:
        heatmap_data = top_results.values[:20].reshape(4, 5)
        heatmap_labels = np.array(top_results.index[:20]).reshape(4, 5)

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=heatmap_labels, fmt="", cmap="rocket_r",
                    cbar_kws={'label': 'Probability Score'}, linewidths=.5)
        plt.title("Interaction Risk Heatmap (Top 20 Drugs)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough predictions to generate full 4x5 heatmap.")
except ValueError as e:
    print(f"Could not generate heatmap: {e}")

# 3. Feature Importance
importances = np.mean([estimator.feature_importances_ for estimator in model.estimators_], axis=0)
feat_importances = pd.Series(importances, index=X.columns)
top_features = feat_importances.nlargest(15)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='teal')
plt.title("Feature Importance (Aggregated across all targets)")
plt.xlabel("Mean Importance Score")
plt.gca().invert_yaxis() # Highest importance on top
plt.tight_layout()
plt.show()
