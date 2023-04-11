# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from random import randint
from datetime import datetime
from datetime import datetime, timedelta

# function to run the kmeans clustering

def kmeans_clustering(file_path):
    # Load the dataset
    df = pd.read_csv('tracklist.csv')

    # Preprocessing for k-means
    df_cl = df[['tempo', 'loudness', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence']]
    df_cl = df_cl.replace(0, 0.1)
    df_cl = df_cl.fillna(df_cl.mean())

    # Log transformation
    df_log = np.log(df_cl)

    # Standardization
    std_scaler = StandardScaler()
    df_scaled = std_scaler.fit_transform(df_cl)

    # Min Max Scaling
    scaler = MinMaxScaler()
    df_scaled_positive = scaler.fit_transform(df_log)

    # Kmeans
    model = KMeans(n_clusters=10, random_state=42)
    model.fit(df_scaled)
    df = df.assign(KMeans=model.labels_)

    # Rename ClusterLabel to KMeans
    df = df.rename(columns={'ClusterLabel': 'KMeans'})

    # Cluster Label to categorical
    df['KMeans'] = df['KMeans'].astype('category')

    # Save the dataframe to csv
    df.to_csv('tracklist_kmeans.csv', index=False)

    return df

# kmeans_clustering('tracklist.csv')


# Define the default_args for the DAG
default_args = {
    'owner': 'your_name',  # Replace with your name
    'start_date': datetime(2023, 1, 1),  # Replace with the start date of your DAG
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG with the default_args
dag = DAG(
    'kmeans_clustering_dag',  # Replace with the name of your DAG
    default_args=default_args,
    schedule_interval='@daily',  # Replace with the desired schedule interval for your DAG
)

# Define the PythonOperator to run the kmeans_clustering() function
kmeans_task = PythonOperator(
    task_id='kmeans_clustering_task',  # Replace with the name of the task
    python_callable=kmeans_clustering,  # Replace with the actual name of your function
    op_args=['tracklist.csv'],  # Replace with the argument(s) to pass to your function
    dag=dag,
)