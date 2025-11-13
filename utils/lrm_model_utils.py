import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# Load dataset
df = pd.read_csv('/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/faers_with_embeddings_ready.csv')
print('CSV Read Successfully')

# Identify columns
reaction_cols = [c for c in df.columns if c.startswith('reaction_')]
drug_cols = [c for c in df.columns if c.startswith('drug_')]
demo_cols = ['age', 'sex_1', 'sex_2']

glp1_drugs = [d for d in [
    'drug_Liraglutide',
    'drug_Semaglutide',
    'drug_Dulaglutide',
    'drug_Exenatide'
] if d in df.columns]

# Define X and y
X = df[reaction_cols + demo_cols + glp1_drugs]
y = df[[c for c in drug_cols if c not in glp1_drugs]]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
print('Split complete')

# Drop constant columns from y_train
constant_cols = [col for col in y_train.columns if y_train[col].nunique() < 2]
if constant_cols:
    print(f"Dropping {len(constant_cols)} constant columns with only one class: {constant_cols[:5]}...")
    y_train = y_train.drop(columns=constant_cols)
    y_test = y_test.drop(columns=constant_cols, errors='ignore')

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Scaled Train and Test Split Complete')

# Train model
base_model = LogisticRegression(
    max_iter=1000,     # more iterations
    solver='saga',
    penalty='l1',
    tol=1e-2,          # easier stopping criteria
    n_jobs=-1
)

model = MultiOutputClassifier(base_model)
model.fit(X_train_scaled, y_train)
print('Model trained successfully!')

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Example user input
user_input = {
    'age': 55,
    'sex_1': 1,
    'sex_2': 0,
    'drug_Liraglutide': 1,
    'drug_Semaglutide': 0,
    'drug_Dulaglutide': 0,
    'drug_Exenatide': 0,
    'reaction_nausea': 1,
    'reaction_vomiting': 1,
    'reaction_dizziness': 1,
    'reaction_headache': 0,
}

user_df = pd.DataFrame([user_input], columns=X.columns).fillna(0)
user_scaled = scaler.transform(user_df)

# Predict probabilities
y_prob = np.column_stack([
    clf.predict_proba(user_scaled)[:, 1] for clf in model.estimators_
])

drug_probabilities = pd.Series(y_prob[0], index=y_train.columns)
top_drugs = drug_probabilities.sort_values(ascending=False).head(10)

print("\nTop 10 Most Likely Interacting Drugs:")
print(top_drugs)

