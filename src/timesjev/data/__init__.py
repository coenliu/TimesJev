from timesjev.data.tsqa import load_tsqa

__all__ = ["load_tsqa", "load_samples"]


def load_samples(data_cfg: dict):
    """Load the TSQA samples an experiment config asks for."""
    if data_cfg.get("source", "tsqa") != "tsqa":
        raise ValueError(f"unknown data source: {data_cfg['source']}")
    samples = load_tsqa(
        converters=data_cfg["converters"],
        splits=data_cfg.get("splits", ["train"]),
        val_fraction=data_cfg.get("val_fraction", 0.02),
        shuffle=data_cfg.get("shuffle_options", True),
        seed=data_cfg.get("seed", 0),
        limit=data_cfg.get("limit"),
        domains=data_cfg.get("domains"),
        max_per_domain=data_cfg.get("max_per_domain"),
        domain_offset=data_cfg.get("domain_offset", 0),
        offset_scope=data_cfg.get("offset_scope", "domain"),
    )
    if data_cfg.get("ids_file"):        # keep only the ids of a fixed subset (configs/subsets/*.json, "ids": {key: [...]})
        import json
        from timesjev.config import PROJECT_ROOT
        keep = set(json.loads((PROJECT_ROOT / data_cfg["ids_file"]).read_text())["ids"][data_cfg["ids_key"]])
        samples = [s for s in samples if s.id in keep]
    return samples
