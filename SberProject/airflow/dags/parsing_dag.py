import airflow
import pendulum
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": pendulum.today("Europe/Moscow"),
}

dag = DAG("parsing-dag",
          default_args=default_args,
          schedule_interval=timedelta(minutes=60),
          catchup=False,
          is_paused_upon_creation=False,
          tags=["parsing"]
          )

resources = k8s.V1ResourceRequirements(
    requests={
        "memory": "512Mi",
        "cpu": 0.5
    },
    limits={
        "memory": "4Gi",
        "cpu": 1.0
    }
)

ParsingTask = KubernetesPodOperator(
    namespace="airflow",
    image="cr.yandex/crpl2ivjm7kaokv219ge/parsing_service:latest",
    labels={"app": "parsing-task"},
    env_vars={"CPU_CORES": "4", "WEAVIATE_HOST": "10.2.197.184", "WEAVIATE_PORT": "80"},
    name="parsing-task",
    task_id="parsing-task-id",
    startup_timeout_seconds=30,
    resources=resources,
    get_logs=True,
    dag=dag,
    log_events_on_failure=True,
    is_delete_operator_pod=True)


ParsingTask
