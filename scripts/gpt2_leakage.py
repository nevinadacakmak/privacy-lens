# gpt2_leakage.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Canary = unique string we insert into training data
CANARY = "NevinAda123"

def build_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def main():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Tiny dataset with repeated canary
    with open("train.txt", "w") as f:
        for _ in range(200):
            f.write(f"My secret code is {CANARY}.\n")
            f.write("The weather is nice today.\n")

    train_dataset = build_dataset("train.txt", tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./gpt2-output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=200,
        save_total_limit=1,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Test leakage: prompt with prefix and see if canary is generated
    prompt = "My secret code is NevinAda"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5, num_return_sequences=3)

    print("Leakage test outputs:")
    for o in outputs:
        print(tokenizer.decode(o, skip_special_tokens=True))

if __name__ == "__main__":
    main()
