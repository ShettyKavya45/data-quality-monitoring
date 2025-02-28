import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Initialize Flask App
app = Flask(__name__)

# Configure Folders
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'  # Use static for serving images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load CSV file
        data = pd.read_csv(filepath)

        # Ensure 'target' column exists
        if 'target' not in data.columns:
            return "Error: The dataset must contain a 'target' column."

        # Step 1: Data Cleaning
        ## Handling Missing Values
        imputer = KNNImputer(n_neighbors=5)
        data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        ## Missing Values per Column
        missing_values = data.isnull().sum()
        missing_values_html = missing_values.to_frame().to_html()

        ## Generate Missing Values Bar Chart
        plt.figure(figsize=(8, 4))
        missing_values.plot(kind='bar', color='blue')
        plt.title("Missing Values per Column")
        plt.xlabel("Columns")
        plt.ylabel("Missing Values Count")
        missing_plot_path = os.path.join(STATIC_FOLDER, 'missing_values.png')
        plt.savefig(missing_plot_path)
        plt.close()

        ## Outlier Detection using Isolation Forest
        iso = IsolationForest(contamination=0.01, random_state=42)
        data_cleaned['Anomaly'] = iso.fit_predict(data_cleaned.drop(columns=['target']))
        data_cleaned = data_cleaned[data_cleaned['Anomaly'] == 1].drop(columns=['Anomaly'])

        ## Addressing Data Imbalance using SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(data_cleaned.drop(columns=['target']), data_cleaned['target'])

        # Step 2: Model Training
        ## Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        ## Decision Tree Model
        dt_model = DecisionTreeClassifier(max_depth=3)
        dt_model.fit(X_train, y_train)
        dt_predictions = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(y_test, dt_predictions)

        ## Random Forest Model
        rf_model = RandomForestClassifier(random_state=42, n_estimators=30, max_depth=5, min_samples_split=10, min_samples_leaf=5)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        # Step 3: Model Evaluation
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

        # Generate Confusion Matrix Visualization
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        conf_matrix_path = os.path.join(STATIC_FOLDER, 'confusion_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.close()

        # Duplicate Rows Analysis
        duplicate_rows = data.duplicated().sum()

        ## Duplicate Rows Distribution Plot
        plt.figure(figsize=(6, 4))
        sns.histplot(data.duplicated(), bins=3, kde=True)
        plt.title("Duplicate Rows Distribution")
        duplicate_plot_path = os.path.join(STATIC_FOLDER, 'duplicate_distribution.png')
        plt.savefig(duplicate_plot_path)
        plt.close()

        # Data Preview (First 5 rows)
        data_preview = data.head().to_html()

        return render_template('dashboard.html', 
                               dt_accuracy=dt_accuracy, 
                               class_report=class_report, 
                               roc_auc=roc_auc, 
                               conf_matrix_path=conf_matrix_path,
                               missing_values_html=missing_values_html,
                               missing_plot_path=missing_plot_path,
                               duplicate_rows=duplicate_rows,
                               duplicate_plot_path=duplicate_plot_path,
                               data_preview=data_preview)

if __name__ == '__main__':
    app.run(debug=True)
