from collaborative.nodes.monitoring_nodes import MonitoringNodes


def run(artifact_path: str = "predictions.json"):
    monitoring_nodes = MonitoringNodes()
    train_data = monitoring_nodes.get_training_data(artifact_path)
    prod_data = monitoring_nodes.get_production_data()
    monitoring_nodes.report_metrics(prod_data, train_data)
    monitoring_nodes.get_metrics_from_drift_report()
    monitoring_nodes.send_to_prometheus()
