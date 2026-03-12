from .adapters import BaseTopicModel, ECRTM, ETM, LossBundle, NSTM, NVDM, PLDA, SCHOLAR


MODEL_REGISTRY = {
    "ecrtm": ECRTM,
    "etm": ETM,
    "nstm": NSTM,
    "nvdm": NVDM,
    "plda": PLDA,
    "scholar": SCHOLAR,
}


def list_supported_topic_models():
    return sorted(MODEL_REGISTRY.keys())


def create_topic_model(model_name, **model_kwargs):
    normalized_name = model_name.strip().lower()
    if normalized_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized_name](**model_kwargs)
    raise ValueError(
        "Unsupported topic model '{}'. Currently available: {}".format(
            model_name,
            list_supported_topic_models(),
        )
    )


__all__ = [
    "BaseTopicModel",
    "LossBundle",
    "ECRTM",
    "ETM",
    "NSTM",
    "NVDM",
    "PLDA",
    "SCHOLAR",
    "create_topic_model",
    "list_supported_topic_models",
]
