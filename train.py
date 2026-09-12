from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
import json
from scipy.stats import spearmanr
from tqdm import tqdm
import gc
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import random
from utils import *
from models import *

device = torch.device("cuda")

wiki_dataset = load_dataset(
    "text", 
    data_files={"train": "https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt"}
)
corpus = wiki_dataset["train"].to_list()


def evaluate_sts(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch1, batch2, targets in dataloader:
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            batch2 = {k: v.to(device) for k, v in batch2.items()}
            targets = targets.to(device)

            emb1 = model(input_ids=batch1["input_ids"], attention_mask=batch1["attention_mask"])
            emb2 = model(input_ids=batch2["input_ids"], attention_mask=batch2["attention_mask"])
            
            predictions = F.cosine_similarity(emb1, emb2, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    spearman_corr, _ = spearmanr(all_preds, all_targets)
    return spearman_corr


def run_val(model, tokenizer, batch_size = 128):
    model.eval()
    for y in ["12", "13", "14", "15", "16"]:
        dataset = load_sts12_16_dataset(years = [y])
        
        collator = STSCollator(tokenizer=tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        
        score = evaluate_sts(model, dataloader, device = torch.device("cuda"))
        print(f"STS{y}: {score:.6f}")


def config_to_model(
        model_name,
        hidden_dim,
        n_layers = None,
        n_heads = None,
        mode = "hlc",
    ):
    model = None
    
    if mode == "hlc":
        model = HLCModel(
            model_name = model_name,
            hidden_dim = hidden_dim,
            n_layers = n_layers,
            n_heads = n_heads,
        ).to(device)
    elif mode == "baseline":
        model = BaselineModel(
            model_name = model_name,
            hidden_dim = hidden_dim,   
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def train_model(model, tokenizer, batch_size = 128, epochs = 1, log_steps = 100, name = "name"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn(batch):
        return tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=64, 
            return_tensors="pt"
        )

    # data
    dataset = UnsupervisedDataset(corpus)
        
    g = torch.Generator()
    g.manual_seed(22022009)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=g)
    
    epochs = epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = int(total_steps * 0.05), 
        num_training_steps = total_steps
    )
    
    loss_fn = SimCSELoss(temperature=0.05)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_s = 0

        for step, batch in tqdm(enumerate(train_dataloader, 1), desc = f"ep [{epoch+1}/{epochs}]", total=len(train_dataloader)):
            num_s += batch["input_ids"].shape[0]
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            emb1 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            emb2 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            loss = loss_fn(emb1, emb2)
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % log_steps == 0:
                tqdm.write(f"Step {step}: loss = {total_train_loss / num_s:.8f}")
                total_train_loss = 0.0
            
            print("Validating on sts:")
            run_val(model, tokenizer, batch_size)
        
    torch.save(model.state_dict(), f"{name}_model.pt")

    del dataset
    gc.collect()
    torch.cuda.empty_cache()