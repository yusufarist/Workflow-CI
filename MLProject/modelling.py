import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import os
import dagshub
import argparse

# Initialize DagsHub
dagshub.init(repo_owner='yusufarist', repo_name='Eksperimen_SML_Yusuf-Arist', mlflow=True)

# Set MLflow tracking URI ke DagsHub
mlflow.set_tracking_uri("https://dagshub.com/yusufarist/Eksperimen_SML_Yusuf-Arist.mlflow")

def load_data(data_dir):
    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train_model(data_dir):
    # Load data
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Start MLflow run
    with mlflow.start_run(run_name="heart_disease_prediction_advanced"):
        # Initialize model
        model = RandomForestClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "specificity": recall_score(y_test, y_pred, pos_label=0),
            "balanced_accuracy": (recall_score(y_test, y_pred) + recall_score(y_test, y_pred, pos_label=0)) / 2
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save feature importance to CSV
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        })
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        # Save confusion matrix to CSV
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                           index=['Actual Negative', 'Actual Positive'],
                           columns=['Predicted Negative', 'Predicted Positive'])
        cm_df.to_csv('confusion_matrix.csv')
        mlflow.log_artifact('confusion_matrix.csv')
        
        # Save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report.csv')
        mlflow.log_artifact('classification_report.csv')
        
        # Print results
        print("Best parameters:", grid_search.best_params_)
        print("\nMetrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="heart_preprocessing")
    args = parser.parse_args()
    
    train_model(args.data_dir)