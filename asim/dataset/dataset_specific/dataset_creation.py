from omegaconf import OmegaConf


def dataset_creation(cfg: OmegaConf) -> None:
    """
    Creates the dataset based on the provided configuration.
    :param cfg: OmegaConf. Configuration that is used to run the experiment.
    """
    from asim.dataset.dataset_specific.carla.data_conversion import CarlaDataset
    from asim.dataset.dataset_specific.nuplan.data_conversion import NuPlanDataset

    if cfg.dataset.name == "carla":
        dataset = CarlaDataset(cfg.output_path, cfg.dataset.split)
    elif cfg.dataset.name == "nuplan":
        dataset = NuPlanDataset(cfg.output_path, cfg.dataset.split)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    dataset.convert(cfg.log_name)
