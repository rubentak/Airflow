
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import os

# Define the kmeans_clustering() function
def kmeans_clustering():
    # Load the dataset
    df = pd.read_csv('/Users/erictak/PycharmProjects/Airflow/tracklist.csv')

    # Preprocessing for k-means
    df_cl = df[['tempo', 'loudness', 'danceability', 'energy', 'key', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence']]
    df_cl = df_cl.replace(0, 0.1)
    df_cl = df_cl.fillna(df_cl.mean())

    # Standardization
    std_scaler = StandardScaler()
    df_scaled = std_scaler.fit_transform(df_cl)

    # Kmeans
    model = KMeans(n_clusters=10, random_state=42)
    model.fit(df_scaled)
    df = df.assign(KMeans=model.labels_)

    # Rename ClusterLabel to KMeans
    df = df.rename(columns={'ClusterLabel': 'KMeans'})

    # Cluster Label to categorical
    df['KMeans'] = df['KMeans'].astype('category')

    # Save the dataframe to csv
    save_path = os.path.join('/Users/erictak/airflow', 'tracklist_kmeans.csv')
    df.to_csv('save_path', index=False)

    return print(df.head())

#%%

# Define the default_args for the DAG
default_args = {
    'owner': 'your_name',  # Replace with your name
    'start_date': datetime(2023, 4, 11),  # Replace with the start date of your DAG
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG with the default_args
dag = DAG(
    'dag_2_kmeans',  # Replace with the name of your DAG
    default_args=default_args,
    schedule_interval='@hourly',  # Replace with the desired schedule interval for your DAG
)

# Define the PythonOperator to run the kmeans_clustering() function
kmeans_task = PythonOperator(
    task_id='kmeans_clustering_task',  # Replace with the name of the task
    python_callable=kmeans_clustering,  # Replace with the actual name of your function
    dag=dag,
)
