import logging

import numpy as np
import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)

DATASET_PATH = "data/amazon/amazon_top500.csv"
df = pd.read_csv(DATASET_PATH)

df_str = "\n".join(df[df.columns].apply(lambda x: ",".join(x.astype(str)), axis=1))

with open("data/amazon/amazon_top500.txt", "w") as f:
    f.write(df_str)


with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = f.read()

n = len(data)
logger.info(f"Data has {n:,} characters")
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

logger.info(f"train has {len(train_ids):,} tokens")
logger.info(f"val has {len(val_ids):,} tokens")

logger.info("Exporting to bin files")
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile("data/amazon/train.bin")
val_ids.tofile("data/amazon/val.bin")
