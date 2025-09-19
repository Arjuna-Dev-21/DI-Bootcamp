import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

df = pd.read_csv("heart_disease_uci.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.drop(columns=["slope", "ca", "thal"], errors="ignore")

df.hist(figsize=(12, 8))
plt.suptitle("Distribution of Numerical Features")
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.title("Boxplot of Features")
plt.xticks(rotation=90)
plt.show()

numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print(df['num'].value_counts())
sns.countplot(x='num', data=df)
plt.title("Target Distribution (0–4)")
plt.show()


X = df.drop(columns=["num"])
y = (df["num"] > 0).astype(int)

categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if col not in categorical_features]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ]), numerical_features),
        
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

log_reg = LogisticRegression(max_iter=1000)

param_grid = [
    {"clf__penalty": ["l1"], "clf__solver": ["liblinear", "saga"], "clf__C": [0.01, 0.1, 1, 10]},
    {"clf__penalty": ["l2"], "clf__solver": ["liblinear", "saga"], "clf__C": [0.01, 0.1, 1, 10]}
]

pipe = Pipeline(steps=[("preprocessor", preprocessor),
                       ("clf", log_reg)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=1, error_score="raise")
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
