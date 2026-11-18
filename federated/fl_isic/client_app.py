"""Clients execute server-authored initialization and named round requests."""
from dataclasses import dataclass

from torch import nn
from isic_dataloaders import ISICDataset

import mlflow
from mlflow.tracking import MlflowClient
from flwr.app import ArrayRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from fl_isic.protocol import (ClientIdentity, ClientInitialization, RoundRequest,
                             ClientResponse, InitializationAcknowledgement)
from fl_isic.task import (CLIENT_METRICS, build_model, class_counts, dataset_from_references,
                         evaluate as evaluate_model, local_train, make_loader, scalar_metrics)
from fl_isic.federated_plotter import FederatedPlotter
from net_training.training_configuration import TrainingCFG, set_seed

app = ClientApp()


@app.query("identity")
def identity(msg: Message, context: Context) -> Message:
    return Message(content=RecordDict({"identity": ClientIdentity.from_context(context).to_record()}),
                   reply_to=msg)


@app.query("initialize")
def initialize(msg: Message, context: Context) -> Message:
    initialization = ClientInitialization.from_record(msg.content["initialization"])
    if initialization.identity != ClientIdentity.from_context(context):
        raise ValueError("Initialization was addressed to a different client")
    cfg = TrainingCFG(initialization.training_config)
    train_data = dataset_from_references(initialization.data_assignment.train, cfg.train_transforms)
    val_data = dataset_from_references(initialization.data_assignment.val, cfg.val_transforms)
    mlflow.set_tracking_uri(initialization.tracking.uri)
    client = MlflowClient()
    distributions = {"train": class_counts(train_data), "val": class_counts(val_data)}
    client.log_dict(initialization.tracking.run_id, distributions, "data_distribution.json")
    for split, counts in distributions.items():
        figure = FederatedPlotter.plot_distribution({str(initialization.identity.client_id): counts}, split)
        try:
            client.log_figure(initialization.tracking.run_id, figure, f"data_distribution/{split}.png")
        finally:
            figure.clear()
    initialization.save(context)
    ack = InitializationAcknowledgement(initialization.identity.client_id,
                                        initialization.data_assignment.dataset_version)
    return Message(content=RecordDict({"acknowledgement": ack.to_record()}), reply_to=msg)


@dataclass
class PreparedClientTask:
    initialization: ClientInitialization
    request: RoundRequest
    cfg: TrainingCFG
    model: nn.Module
    train_data: ISICDataset
    val_data: ISICDataset

    @property
    def label(self):
        return f"Hospital {self.request.client_id} | round {self.request.server_round}"


def _prepare(msg: Message, context: Context) -> PreparedClientTask:
    initialization = ClientInitialization.load(context)
    request = RoundRequest.from_message(msg)
    if request.client_id != initialization.identity.client_id:
        raise ValueError("Round request was addressed to a different client")
    cfg = TrainingCFG(initialization.training_config)
    set_seed(request.seed, pipelines=(cfg.train_transforms, cfg.val_transforms))
    model = build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return PreparedClientTask(initialization, request, cfg, model,
        dataset_from_references(initialization.data_assignment.train, cfg.train_transforms),
        dataset_from_references(initialization.data_assignment.val, cfg.val_transforms))


@app.train()
def train(msg: Message, context: Context) -> Message:
    task = _prepare(msg, context)
    request, cfg, model = task.request, task.cfg, task.model
    train_data, val_data, label, seed = task.train_data, task.val_data, task.label, request.seed
    print(f"{label}: training on {len(train_data)} images (mu={request.proximal_mu})")
    mlflow.set_tracking_uri(task.initialization.tracking.uri)
    run_id = task.initialization.tracking.run_id
    tracking_client = MlflowClient()
    client_id = request.client_id
    distributions = {"train": class_counts(train_data), "val": class_counts(val_data)}
    histories = {key: {point.step: point.value for point in
                 tracking_client.get_metric_history(run_id, f"train/{key}")}
                 for key in ("accuracy", "loss")}
    # Append steps even when an interrupted round is retried.
    history = tracking_client.get_metric_history(run_id, "local_epoch")
    epoch_offset = max((point.step for point in history), default=0)

    def log_epoch(epoch, result):
        step = epoch_offset + epoch
        values = result.scalar_metrics()
        for key, value in values.items():
            if isinstance(value, (int, float)):
                tracking_client.log_metric(run_id, f"train/{key}", float(value), step=step)
        tracking_client.log_metric(run_id, "server_round", request.server_round, step=step)
        tracking_client.log_metric(run_id, "local_epoch", epoch, step=step)
        for key in histories:
            if key in values:
                histories[key][step] = float(values[key])
        figure = FederatedPlotter.plot_client_training(client_id, histories)
        try:
            tracking_client.log_figure(run_id, figure, "training_metrics.png")
        finally:
            figure.clear()

    metrics = local_train(
        model,
        make_loader(train_data, cfg, shuffle=True, seed=seed),
        cfg,
        epochs=request.local_epochs,
        mu=request.proximal_mu,
        label=label,
        on_epoch=log_epoch,
    )
    validation = scalar_metrics(evaluate_model(model,
        make_loader(val_data, cfg, shuffle=False, seed=seed), cfg,
        f"{label} - local model validation"), CLIENT_METRICS)
    for key, value in validation.items():
        tracking_client.log_metric(run_id, f"val/{key}", value,
                                   step=epoch_offset + request.local_epochs)
    response = ClientResponse(client_id, request.server_round, len(train_data),
        scalar_metrics(metrics, CLIENT_METRICS), distributions,
        ArrayRecord(model.cpu().state_dict()), validation, len(val_data))
    return Message(content=response.to_content(), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    task = _prepare(msg, context)
    metrics = evaluate_model(task.model,
        make_loader(task.val_data, task.cfg, shuffle=False, seed=task.request.seed),
        task.cfg, f"{task.label} - local validation")
    response = ClientResponse(task.request.client_id, task.request.server_round,
                              len(task.val_data), scalar_metrics(metrics, CLIENT_METRICS))
    return Message(content=response.to_content(), reply_to=msg)
