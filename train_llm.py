from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import time

start_time = time.time()





tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
model=GPT2LMHeadModel.from_pretrained("gpt2")




dataset_training = TextDataset(

tokenizer=tokenizer,
file_path="./dataset.txt",
block_size=16

)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="./trained_data",
    per_device_train_batch_size=1,
    save_steps=1,
    save_total_limit=1,
    overwrite_output_dir=True,
    num_train_epochs=6
    
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_training,
)

trainer.train()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")
with open("training_time.txt", "w") as f:
    f.write(f"Training took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).\n")



trainer.save_model("./trained_data")
tokenizer.save_pretrained("./trained_data")