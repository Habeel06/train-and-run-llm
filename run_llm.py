from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch


model_dest="./trained_data"

tokenizer=GPT2Tokenizer.from_pretrained("./trained_data")
model=GPT2LMHeadModel.from_pretrained("./trained_data")


model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def conversation():
    print("Hello there fellow stranger.I am Sunok! How can I help?")

    while True:
        
        a=input("You:")
        
        if a== "exit":
            break

        usr_promt_token_process_fr = tokenizer.encode(a, return_tensors="pt").to(device)
        # # usr_promt_token_process_fr_ids=usr_promt_token_process_fr[' usr_promt_token_process_fr_ids'].to(device)
        # attention_mask=usr_promt_token_process_fr['attention_mask'].to(device)
        res = model.generate(
            usr_promt_token_process_fr, 
            max_length=100, 
            #  usr_promt_token_process_fr_ids= usr_promt_token_process_fr_ids,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            # attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
           
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7
        )
        
    
        response = tokenizer.decode(res[0], skip_special_tokens=True)
    
        print("Bot:", response[len(a):].strip())

if __name__ == "__main__":
    conversation()