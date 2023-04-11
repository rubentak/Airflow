# Final assignment Big Data Infrastructure 

## By Ruben Tak

### Assignment:

Create an Airflow pipeline for an end-to-end ML workflow: 

* get some data in bronze(s3)  (raw csv/json...)
* Join it with other data and save it in silver (s3) parquet and with the splits and format needed for training a model
* Execute a training job with the data
* Track parameters and metrics (e.g. in logs)
* Deploy the model to production as an endpoint/API container or execute a batch prediction on some data
* TIP: Don't reinvent the wheel, the model does not need to be complex. You can use pre-created ones

You can run it in AWS, on your local PC or hybrid

##### Deliver

* Architecture documentation (2-3 pages)
* Purpose and explanation of the solution and its components
* Architecture diagram
* All the code
* Proof of the things running (pictures)


### What is airflow?
Airflow in Python refers to an open-source platform used for orchestrating complex workflows and data pipelines. It provides a framework for defining, scheduling, and monitoring tasks as directed acyclic graphs (DAGs), which allows users to specify how tasks are organized and executed. Airflow is often used in data engineering and data science pipelines to automate workflows that involve multiple steps or dependencies, such as data ingestion, data processing, and data transformation. Airflow provides a web-based user interface for visualizing and managing workflows, along with a rich set of operators and sensors that can be used to define tasks and their dependencies. It also supports advanced features such as dynamic task generation, retries, and error handling, making it a powerful tool for managing complex data workflows in Python.

### Steps:


#### Step 1: Set up Airflow

Install Apache Airflow on your local PC or on an AWS EC2 instance.
Create an Airflow DAG (Directed Acyclic Graph) to define the workflow.

#### Step 2: Data Ingestion

Use Airflow to trigger a data ingestion task that retrieves the raw data from S3 (bronze).
Perform any necessary data cleaning and preprocessing.
Save the cleaned data to a new location in S3 (silver) in Parquet format, along with the splits and format needed for model training.

#### Step 3: Model Training

Create a training job using a machine learning library or framework of your choice (e.g., scikit-learn, TensorFlow, PyTorch) to train a model using the cleaned data in S3 (silver).
Log relevant parameters and metrics (e.g., hyperparameters, loss, accuracy) during the training process using a logging library (e.g., MLflow, TensorBoard).

#### Step 4: Model Deployment

Deploy the trained model to production as an endpoint or API container using a containerization tool like Docker.
Alternatively, if you want to execute batch predictions on some data, use Airflow to trigger a batch prediction task that uses the trained model to make predictions on new data.

#### Step 5: Monitoring and Logging

Use Airflow to set up monitoring tasks that periodically check the status of the pipeline components (e.g., data ingestion, model training, model deployment).
Use logging libraries or tools (e.g., ELK stack, CloudWatch, Splunk) to capture and analyze logs from the pipeline components for troubleshooting, auditing, and performance monitoring purposes.

#### Step 6: Documentation and Proof of Execution

Create architecture documentation that includes the purpose and explanation of the solution and its components, along with an architecture diagram that illustrates the flow of data and tasks in the pipeline.
Include all the code used in the pipeline, including the Airflow DAG definition, data ingestion, model training, and model deployment code.
Provide proof of execution, such as screenshots or output logs, to demonstrate that the pipeline is running and producing the expected results.

Note: Depending on your specific use case and environment (AWS, local PC, hybrid), you may need to configure additional components such as AWS S3, AWS SageMaker, or Docker in your pipeline.

Once you have completed the above steps, you will have a functional Airflow pipeline for an end-to-end ML workflow, including data ingestion, model training, model deployment, and monitoring.

### steps so far:

- created a kmeans ML function
- defined the DAG in airflow
- run the DAG in airflow
- 

## references:

What is DAG?
https://www.youtube.com/watch?v=1Yh5S-S6wsI
