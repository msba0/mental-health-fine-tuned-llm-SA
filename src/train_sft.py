import argparse, json
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq,
)

import torch


@dataclass
class Config:
    model_name: str
    output_dir: str
    train_file: str
    eval_file: str
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    logging_steps: int = 20
    save_steps: int = 200
    bf16: bool = True
    report_to: str = "tensorboard"


def load_config(path: str) -> Config:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Config(**d)


def read_jsonl(path):
    for line in open(path, "r", encoding="utf-8"):
        yield json.loads(line)


def build_prompt(user_text: str) -> str:
    return (
        "You are an empathetic counselor. "
        "Respond concisely and supportively.\n\n"
        f"User: {user_text}\n\nAssistant:"
    )


def load_ds(train_path, eval_path, tok):
    def encode_mask(ex):
        user = ex["input"]
        ans  = ex["output"]

        prompt = build_prompt(user)
        full_text = (prompt + " " + ans).strip()

        full = tok(full_text, truncation=True, max_length=512)
        prompt_ids = tok(prompt, truncation=True, max_length=512)["input_ids"]

        labels = full["input_ids"][:] 
        mask_len = min(len(prompt_ids), len(labels))
        for i in range(mask_len):
            labels[i] = -100

        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    train_list = [encode_mask(x) for x in read_jsonl(train_path)]
    eval_list  = [encode_mask(x) for x in read_jsonl(eval_path)]

    return Dataset.from_list(train_list), Dataset.from_list(eval_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_qwen.yml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    cfg.per_device_train_batch_size = int(cfg.per_device_train_batch_size)
    cfg.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
    cfg.num_train_epochs = float(cfg.num_train_epochs)
    cfg.learning_rate = float(cfg.learning_rate)

    use_cuda = False
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            use_cuda = True
        except Exception as e:
            print("CUDA detected but not usable, falling back to CPU:", e)

    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if (cfg.bf16 and use_cuda) else None

    print("== Run plan ==")
    print("Model:", cfg.model_name)
    print("Train file:", cfg.train_file)
    print("Eval file:", cfg.eval_file)
    print("Device:", device, "bf16:", bool(dtype))

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype).to(device)

    train_data, eval_data = load_ds(cfg.train_file, cfg.eval_file, tok)

    args_tr = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=cfg.bf16 and use_cuda,
        no_cuda=not use_cuda,  
        report_to=cfg.report_to.split(",") if cfg.report_to else None,
    )

    data_collator = DataCollatorForSeq2Seq(
    tokenizer=tok,
    model=model,
    padding="longest",
    label_pad_token_id=-100,
    return_tensors="pt",
    )


    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        tokenizer=tok,
    )

    trainer.train()
    print("Done. Checkpoints in:", cfg.output_dir)
    
    
    from pathlib import Path
    final_dir = Path(cfg.output_dir) / "checkpoint-last"
    trainer.save_model(final_dir)     
    tok.save_pretrained(final_dir)    
    print("Saved final checkpoint to:", final_dir)



if __name__ == "__main__":
    main()
