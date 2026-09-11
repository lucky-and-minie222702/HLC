from train import *
import sys

model_config = [
    {
       "model_name": "google-bert/bert-base-uncased", 
        "n_heads": 12,
        "n_layers": 12,
        "hidden_dim": 768,
    },

    {
       "model_name": "google-bert/bert-base-cased", 
        "n_heads": 12,
        "n_layers": 12,
        "hidden_dim": 768,
    },
    
    {
       "model_name": "google-bert/bert-large-uncased", 
        "n_heads": 16,
        "n_layers": 24,
        "hidden_dim": 1024,
    },

    {
       "model_name": "google-bert/bert-base-cased", 
        "n_heads": 16,
        "n_layers": 24,
        "hidden_dim": 1024,
    },

    {
       "model_name": "FacebookAI/roberta-large",
        "n_heads": 16,
        "n_layers": 24,
        "hidden_dim": 1024,
    },
]

setup = None
with open(sys.argv[1], 'r', encoding='utf-8') as file:
    setup = json.load(file)
    
i = setup["i"]
mode = setup["mode"]
batch_size = setup["batch_size"]
epochs = setup["epochs"]
log_steps = setup["log_steps"]

c = model_config[0]
print(c, mode)
model, tokenizer = config_to_model(**c, mode = mode)
train_model(model, tokenizer, batch_size = batch_size, epochs = epochs, log_steps = log_steps, name = f"{c['model_name'].split('/')[-1]}-{mode}")

# eval
print(c, mode)
for y in ["12", "13", "14", "15", "16"]:
    raw_dataset = load_sts12_16_dataset()
    dataset = load_sts12_16_dataset(years = [y])
    
    collator = STSCollator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    score = evaluate_sts(model, dataloader, device = torch.device("cuda"))
    print(f"STS{y}: {score:.6f}")