## Attention!! This model size is 20gb. You must minimla 48 GB Ram of Computer For Run This Codes!!!

# GPT_UZ

THIS IS A GPT_UZ PROJECT. FOR  BEGIN AI PROGRAMMING, THIS AI MAIN CODE FOR RUN LOCAL HOST!!!
IF YOU KNOW PYTHON PROGRAMMING LANGUAGE! AND YOU WANT RUN THIS PROJECT, SO YOU MUST DO THIS:
TASK 1 
YOU MUST INSTALL THIS LIBARYS:
```pip3 install pytorch```
```pip3 install transformers```
TASK 2 
IF YOU DONE
YOU MUST DO THIS:

~~~from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import os
def load_fine_tuned_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=500, temperature=1.5):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name)
config.max_position_embeddings = 3048
model = GPT2LMHeadModel(config)
model_path = "path/to/path"
fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model(model_path)

while True:
    prompt = input("You:").lower()
    if prompt.lower() == "exit":
        print("AI is Sleeping. Goodbye!")
        break
    else:
        try:
            print(f"Your Question: {prompt}")
            response = generate_response(fine_tuned_model, fine_tuned_tokenizer, prompt, temperature=1.0)
            print("AI:", response)
        except IndexError as err:
            print("Empty question Error")
~~~
