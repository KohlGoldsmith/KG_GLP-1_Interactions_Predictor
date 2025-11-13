import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import numpy as np

# Load dataset
df = pd.read_csv('/Users/kohlgoldsmith/PycharmProjects/Kohl_Goldsmith_GLP-1_Interactions_Predictor/dataset/faers_data/cleaned_data/faers_with_embeddings_ready.csv')
print('CSV Read Successfully')

# Identify columns
reaction_cols = [c for c in df.columns if c.startswith('reaction_')]
drug_cols = [c for c in df.columns if c.startswith('drug_')]
demo_cols = ['age', 'sex_1', 'sex_2']
print('Columns Identified')

# GLP-1 drugs (the user always inputs one of these)
glp1_drugs = [
    'drug_Liraglutide',
    'drug_Semaglutide',
    'drug_Dulaglutide',
    'drug_Exenatide'
]
glp1_drugs = [d for d in glp1_drugs if d in df.columns]  # filter missing ones safely

# Define X and y
X = df[reaction_cols + demo_cols + glp1_drugs]
y = df[[c for c in drug_cols if c not in glp1_drugs]]

# Standard Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
print('Train and test sets allotted')
# Train Random Forest multi-output model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=20, random_state=22))
model.fit(X_train, y_train)
print('Model trained')
# Evaluate
y_pred = model.predict(X_test)
print("\n=== Classification Report (sample metrics) ===")
print(classification_report(y_test, y_pred))

#  Example user input
user_input = {
    # Demographics
    'age': 55,
    'sex_1': 1,  # male
    'sex_2': 0,

    # GLP-1 drug taken
    'drug_Liraglutide': 1,
    'drug_Semaglutide': 0,
    'drug_Dulaglutide': 0,
    'drug_Exenatide': 0,

    # Reported side effects
    'reaction_nausea': 1,
    'reaction_vomiting': 1,
    'reaction_dizziness': 1,
    'reaction_headache': 0,
}

# Setting the user's input (fill missing columns with 0)
user_df = pd.DataFrame([user_input], columns=X.columns).fillna(0)

# Predict probability % instead of just a 0/1
y_prob = np.column_stack([
    clf.predict_proba(user_df)[:, 1] for clf in model.estimators_
])

# Rank top negatively interacting drugs by likelihood
drug_probabilities = pd.Series(y_prob[0], index=y.columns)
top_drugs = drug_probabilities.sort_values(ascending=False).head(10)

print("\nTop 10 Most Likely Interacting Drugs")
print(top_drugs)
