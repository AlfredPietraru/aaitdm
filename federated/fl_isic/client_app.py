"""Flower ClientApp: one hospital that trains and evaluates on its own ISIC images.

Raw images never leave the client; only model weights and metrics are sent back.
"""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_isic.task import (
    CLIENT_METRICS,
    build_model,
    class_counts,
    client_datasets,
    evaluate as evaluate_model,
    load_training_cfg,
    local_train,
    make_loader,
    scalar_metrics,
)
from reproducibility import set_seed

app = ClientApp()


def _prepare(msg: Message, context: Context):
    """Seed, load the received global model, and select this client's partition."""
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    server_round = int(msg.content["config"].get("server-round", 0))
    cfg = load_training_cfg(context.run_config)
    seed = cfg.seed + 1000 * server_round + partition_id
    set_seed(seed, pipelines=(cfg.train_transforms, cfg.val_transforms))
    model = build_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    train_data, val_data = client_datasets(cfg, context.run_config, partition_id, num_partitions)
    label = f"Hospital {partition_id} | round {server_round}"
    # Lets the server keep per-client histories; it strips these before averaging.
    tags = {"partition-id": partition_id, "server-round": server_round}
    return cfg, model, train_data, val_data, label, seed, tags


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg, model, train_data, _, label, seed, tags = _prepare(msg, context)
    mu = float(msg.content["config"].get("proximal-mu", 0.0))
    print(f"{label}: training on {len(train_data)} images {class_counts(train_data)} (mu={mu})")
    metrics = local_train(
        model,
        make_loader(train_data, cfg, shuffle=True, seed=seed),
        cfg,
        epochs=int(context.run_config["local-epochs"]),
        mu=mu,
        label=label,
    )
    reply = RecordDict({
        "arrays": ArrayRecord(model.cpu().state_dict()),
        "metrics": MetricRecord({
            **scalar_metrics(metrics, CLIENT_METRICS),
            "num-examples": len(train_data),
            **tags,
        }),
    })
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    cfg, model, _, val_data, label, seed, tags = _prepare(msg, context)
    metrics = evaluate_model(
        model, make_loader(val_data, cfg, shuffle=False, seed=seed), cfg, f"{label} - local validation",
    )
    reply = RecordDict({
        "metrics": MetricRecord({
            **scalar_metrics(metrics, CLIENT_METRICS),
            "num-examples": len(val_data),
            **tags,
        }),
    })
    return Message(content=reply, reply_to=msg)
