import os, hydra, mlflow, torch, wandb
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer
from src.model import make_model

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    model = make_model(cfg.total_params)
    tok   = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    ds = load_dataset("roneneldan/TinyStories", split="train[:1%]")
    def tokenize(x):
        return tok(x["text"], truncation=True, max_length=256)
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=f"runs/{cfg.total_params}",
        per_device_train_batch_size=cfg.batch,
        learning_rate=cfg.lr,
        max_steps=cfg.steps,
        logging_steps=50,
        save_steps=500,
        report_to=["wandb", "mlflow"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(f"models/{cfg.total_params}")
    mlflow.log_artifact(f"models/{cfg.total_params}")

if __name__ == "__main__":
    main()