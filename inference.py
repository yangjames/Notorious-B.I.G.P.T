#!/usr/bin/env python3

import os
import hashlib

import torch
import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
import logging
import torch.nn.functional as F


dotenv.load_dotenv(override=True)
log = logging.getLogger(__name__)


censor_list = [
    "08a841e996781e9e77d30a4e4420a8f501a280b00624e6d1224bf54aaff73eba",
    "341d56384afc0f47b34ca18273e793be555507a49444c30d3d0588688de46cb3",
    "6ac3c336e4094835293a3fed8a4b5fedde1b5e2626d9838fed50693bba00af0e",
    "d75a838dc758ba17f28bd8dbac605cb70c35465263d5733164521de2f7ef7926",
    "2f5f6ce5ae30b54aa5d7ced1ba566982bab34ba2814a51ce1865d2c2d8815cd4",
    "acd9193a33ddc5c6d96e727c582995490f258bc20ead0eff9cc02d4d089be1fc",
    "85fc17f7069acd39a5c636cd0a6530651096128da447959f5e250824857dc559",
    "d95b1fc7856571b097f46888ec161854b962adf2d1db730fabdbc3a768ad9f72",
    "31506a8448a761a448a08aa69d9116ea8a6cb1c6b3f4244b3043051f69c9cc3c",
    "45fb7c3b72b6856c19294af27110b364b22177ebbae8d7b686378a3daeac678b",
    "5a9d9dcefb56e593cbb6c58eb83a5a112d277053dacf9ffef31449c48bc578bd",
    "0f28c4960d96647e77e7ab6d13b85bd16c7ca56f45df802cdc763a5e5c0c7863",
    "ad505b0be8a49b89273e307106fa42133cbd804456724c5e7635bd953215d92a",
    "30a989afc82c0a21139573591de4e5ff37994f7d1506a9acf2b5997005c2649f",
]
    
@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    # Load data module
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(config.datamodule, _convert_="partial")
    # Create model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model, _convert_="partial")
    # Load checkpoint
    model_path = "pretrained/last.ckpt"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    # Run inference
    with torch.no_grad():
      sample = torch.zeros((1, 1), dtype=torch.long)
      context = torch.randint(low=0, high=config.vocab_size, size=(1, 1), dtype=torch.long)
      num_tokens_to_generate = 1000
      
      for _ in range(num_tokens_to_generate):
          idx_cond = context[:, -config.context_length:]
          # Get the predictions for the current indices
          logits = model.forward(idx_cond)
          # Only generate based on the last context index.
          probs = F.softmax(logits[:, -1, :], dim=1)
          # Sample from the distribution.
          context_next = torch.multinomial(probs, num_samples=1)
          # Append sampled index to the running sequence.
          context = torch.cat((context, context_next), dim = 1)

    generated_lyrics = datamodule._decode(context.tolist()[0])
    censored_lyrics = []
    for word in generated_lyrics:
        word_hash = hashlib.sha256(word.encode("utf-8")).hexdigest()
        if word_hash in censor_list:
            len_middle = len(word) - 2
            censored_word = word[0] + "*" * len_middle + word[-1]
            censored_lyrics += [censored_word]
        else:
            censored_lyrics += [word]
    print(" ".join(censored_lyrics))


if __name__ == "__main__":
    main()
