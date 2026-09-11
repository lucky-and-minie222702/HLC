from train import *

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

i = 0
mode = "hlc"
c = model_config[0]
print(c)
model, tokenizer = config_to_model(**c, mode = mode)
train_model(model, tokenizer, name = f"{c["model_name"].split("/")[-1]}-{mode}")
