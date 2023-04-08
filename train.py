#!/usr/bin/env python3

import os

import torch
import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
import logging


dotenv.load_dotenv(override=True)
log = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    torch.set_num_threads(8)
    pl.seed_everything(config.seed)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule, _convert_="partial")
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model, _convert_="partial")

    callbacks = []
    if "callbacks" in config:
        for _, callback_config in config["callbacks"].items():
            if "_target_" in callback_config:
                log.info(f"Instantiating callback <{callback_config._target_}>.")
                callbacks.append(hydra.utils.instantiate(callback_config, _convert_="partial"))

    loggers = []
    if "loggers" in config:
        for _, logger_config in config["loggers"].items():
            if "_target_" in logger_config:
                log.info(f"Instantiating logger <{logger_config._target_}>.")
                for i in range(10):
                    try:
                        logger = hydra.utils.instantiate(
                            logger_config, _convert_="partial")
                        loggers.append(logger)
                    except:
                        log.info(f"Unable to instantiate logger <{logger_config._target_}>. Retrying...")

    
    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()
