from airflow.models import Variable
from airflow.sensors.base_sensor_operator import BaseSensorOperator
from airflow.utils.decorators import apply_defaults


class FirstTrainSensor(BaseSensorOperator):
    """
    Custom sensor that waits for the First Ever Training Variable to become True.
    """

    @apply_defaults
    def __init__(
        self,
        task_id,
        dag,
        mode="poke",
        timeout=3600,
        *args,
        **kwargs
    ):
        super().__init__(task_id=task_id, mode=mode, timeout=timeout, *args, **kwargs)
        self.task_id = task_id
        self.dag = dag

    def poke(self, context):
        # Get the value of the first ever training to make sure we have predictions
        first_ever_train = Variable.get("first_train_bool", default_var="false")

        # Check if the the value is True
        return first_ever_train.lower() == "true"
