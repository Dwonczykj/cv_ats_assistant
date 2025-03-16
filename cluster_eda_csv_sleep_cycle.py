import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr
import asyncio
import warnings

warnings.filterwarnings("ignore")


async def load_and_preprocess(file_path):
    """Load and preprocess the data."""
    print("Loading and preprocessing data...")
    seps = [";", ",", "|", " ", "\t"]
    for sep in seps: 
        data = pd.read_csv(file_path, sep=sep)
        if data.shape[1] > 1:
            break
    if data.empty:
        raise ValueError("Data is empty or not loaded correctly.")
    # Handle missing values and encode categorical variables
    data.fillna(
        0,
        # method="ffill",
        inplace=True)
    data = pd.get_dummies(data, drop_first=True)
    print("Data loaded and preprocessed.")
    print(f"Columns: {data.columns}")
    return data


async def exploratory_data_analysis(data):
    """Perform exploratory data analysis."""
    print("Performing EDA...")
    eda_results = {
        "summary_statistics": data.describe().to_dict(),
        "correlation_matrix": data.corr().to_dict()
    }
    print("EDA completed.")
    return eda_results


async def clustering_analysis(data):
    """Perform K-Means clustering."""
    print("Performing clustering analysis...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(data)
    data['Cluster'] = clusters
    print("Clustering analysis completed.")
    return data, kmeans.cluster_centers_.tolist()


async def community_analysis(data):
    """Analyze label co-occurrence."""
    print("Performing community analysis...")
    co_occurrence = data.T.dot(data) # BUG: This code is far too slow for large datasets, we need to find a faster way to do this using parallel processing, CUDA and gpu processing
    community_results = {
        "co_occurrence_matrix": co_occurrence.to_dict(),
        "strongest_correlations": {
            f"{col1}-{col2}": pearsonr(data[col1], data[col2])[0]
            for col1 in data.columns for col2 in data.columns if col1 != col2
        }
    }
    print("Community analysis completed.")
    return community_results


async def predictive_modeling(data):
    """Perform predictive modeling for classification and regression."""
    print("Performing predictive modeling...")
    # Prepare the data
    y_class = data['label_to_predict']
    X_class = data.drop(columns=['label_to_predict'])

    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42)

    # Classification model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_class = clf.predict(X_test)
    classification_accuracy = accuracy_score(y_test, y_pred_class)

    # Regression model
    y_reg = data['sleep_score']
    X_reg = data.drop(columns=['sleep_score'])

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg)
    regression_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    print("Predictive modeling completed.")
    return {
        "classification_accuracy": classification_accuracy,
        "regression_rmse": regression_rmse
    }


async def trend_prediction(data, n_days):
    """Predict trends based on the last n days."""
    print("Predicting trends...")
    time_series_data = data[-n_days:]
    trend_results = {
        "optimal_window": n_days,
        "correlations": {
            f"{col1}-{col2}": pearsonr(time_series_data[col1], time_series_data[col2])[0]
            for col1 in time_series_data.columns for col2 in time_series_data.columns if col1 != col2
        }
    }
    print("Trend prediction completed.")
    return trend_results


async def main(file_path):
    data = await load_and_preprocess(file_path)

    eda_task = exploratory_data_analysis(data)
    clustering_task = clustering_analysis(data)
    community_task = community_analysis(data)
    predictive_task = predictive_modeling(data)
    trend_task = trend_prediction(data, n_days=30)

    results = await asyncio.gather(eda_task, clustering_task, community_task, predictive_task, trend_task)

    eda_results, clustered_data, community_results, predictive_results, trend_results = results

    print("\nResults:")
    print("EDA Results:", eda_results)
    print("Clustering Results:", clustered_data)
    print("Community Analysis Results:", community_results)
    print("Predictive Modeling Results:", predictive_results)
    print("Trend Prediction Results:", trend_results)

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(
        sys.argv) > 1 else "/Users/joey/Library/Mobile Documents/com~apple~CloudDocs/SleepCycle/sleepdata2025-01-25.csv"

    asyncio.run(main(file_path))
