import airflow
import pendulum
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": pendulum.today("Europe/Moscow"),
}

with DAG("cifar10-dag",
         default_args=default_args,
         schedule_interval=timedelta(minutes=1200),
         catchup=False,
         is_paused_upon_creation=False,
         tags=["cifar10"]
         ) as dag:

    t1 = BashOperator(
        task_id="download-dataset-id",
        bash_command="python3 ${AIRFLOW_HOME}/../src/load_dataset.py",
    )

    t2 = BashOperator(
        task_id="train-cifar10-id",
        bash_command="MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net python3 ${AIRFLOW_HOME}/../src/train_model.py"
    )

    t1 >> t2
