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

    # Initialize model.
    model = hydra.utils.instantiate(config.model, _convert_="partial").to("cuda")

    x = torch.randint(
        low=0, high=10,
        size=(config.batch_size, config.context_length)
    ).to("cuda")
    y = torch.randint(
        low=0, high=10,
        size=(config.batch_size, config.context_length)
    ).to("cuda")

    # out = model.forward(x)
    # assert out.shape == torch.Size([32, config.context_length, config.vocab_size])

    loss = model.training_step((x, y), 0)
    

if __name__ == "__main__":
    main()
