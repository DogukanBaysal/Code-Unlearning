import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


DATASET_NAME = "dbaysal/KodCode-filtered-2"
MODEL_NAME = "Salesforce/SFR-Embedding-Code-400M_R"
TEXT_COLUMN = "solution"
ID_COLUMN = "id"
QUESTION_COLUMN = "question"

N_SAMPLES = None
BATCH_SIZE = 32
MAX_LENGTH = 8192       # official example uses 8192

OUT_DIR = "embeddings_output"
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = f"{OUT_DIR}/solution_embeddings.npy"
TEXTS_CSV_PATH = f"{OUT_DIR}/solution_texts.csv"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# -----------------------------
# 1. Load dataset
# -----------------------------

ds = load_dataset(DATASET_NAME)

if isinstance(ds, dict):
    split = "train" if "train" in ds else list(ds.keys())[0]
    ds = ds[split]

if N_SAMPLES is not None:
    ds = ds.select(range(min(N_SAMPLES, len(ds))))

texts = ds[TEXT_COLUMN]
texts = ["" if x is None else str(x) for x in texts]

print(ds)
print("Number of examples:", len(ds))
print("Unique texts:", len(set(texts)))


texts_df = pd.DataFrame({
    "embedding_idx": list(range(len(texts))),
    "solution": texts,
    "id": ds[ID_COLUMN],
    "question": ds[QUESTION_COLUMN]
})

texts_df.to_csv(TEXTS_CSV_PATH, index=False)
print("Saved texts CSV:", TEXTS_CSV_PATH)


# -----------------------------
# 2. Load model
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

model.to(device)
model.eval()


# -----------------------------
# 3. Embed using CLS token
# -----------------------------

embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding solutions"):
        batch_texts = texts[i : i + BATCH_SIZE]

        batch_dict = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        outputs = model(**batch_dict)

        # Official-style embedding:
        # take the first token representation
        emb = outputs.last_hidden_state[:, 0]

        # Normalize embeddings
        emb = F.normalize(emb, p=2, dim=1)

        embeddings.append(emb.detach().cpu().float().numpy())

embeddings = np.concatenate(embeddings, axis=0)

print("Embeddings shape:", embeddings.shape)
print("Embedding std:", embeddings.std())
print("NaNs:", np.isnan(embeddings).sum())

np.save(EMB_PATH, embeddings)

print("Saved embeddings:", EMB_PATH)
print("Done.")