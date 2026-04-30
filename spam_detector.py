# ==============================
# SPAM DETECTION PROJECT
# Naive Bayes + Decision Tree
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. LOAD DATASET
# ------------------------------
# You can download dataset from Kaggle: "SMS Spam Collection Dataset"

data = pd.read_csv("books.csv", encoding='latin-1')

# Keep only useful columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# ------------------------------
# 2. SPLIT DATA
# ------------------------------
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 3. TEXT VECTORIZATION (IMPORTANT)
# ------------------------------
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------
# 4. MODEL 1: NAIVE BAYES
# ------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

# ------------------------------
# 5. MODEL 2: DECISION TREE
# ------------------------------
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_vec, y_train)

dt_pred = dt_model.predict(X_test_vec)

# ------------------------------
# 6. EVALUATION
# ------------------------------
print("\n============================")
print("NAIVE BAYES RESULTS")
print("============================")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

print("\n============================")
print("DECISION TREE RESULTS")
print("============================")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# ------------------------------
# 7. TEST YOUR OWN MESSAGE
# ------------------------------
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])

    nb_result = nb_model.predict(msg_vec)[0]
    dt_result = dt_model.predict(msg_vec)[0]

    print("\nMessage:", msg)
    print("Naive Bayes Prediction:", "SPAM" if nb_result else "HAM")
    print("Decision Tree Prediction:", "SPAM" if dt_result else "HAM")

# Example testing
predict_message("Congratulations! You won a free iPhone. Click here now!")
predict_message("Hey, are we meeting tomorrow at 5?")