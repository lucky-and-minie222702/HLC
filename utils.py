from datasets import load_dataset, concatenate_datasets
import torch


def load_sts12_16_dataset(years = ["12", "13", "14", "15", "16"]):
    dataset_list = []
    
    for yr in years:
        ds = load_dataset(f"mteb/sts{yr}-sts")
        for split in ds.keys():
            dataset_list.append(ds[split])

    return concatenate_datasets(dataset_list)


class STSCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sentences1 = [item["sentence1"] for item in batch]
        sentences2 = [item["sentence2"] for item in batch]
        
        # norm
        scores = torch.tensor([item["score"] / 5.0 for item in batch], dtype=torch.float32)

        encoded1 = self.tokenizer(
            sentences1,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoded2 = self.tokenizer(
            sentences2,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return encoded1, encoded2, scores