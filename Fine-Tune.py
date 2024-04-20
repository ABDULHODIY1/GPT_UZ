import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download Model and Tokenizers
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")# Attention!! This model size is 20gb. You must minimla 48 GB Ram of Computer For Run This Codes!!!
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#upload Text Data
with open('yourdataforTXT', 'r') as file:
    texts = file.readlines()  # Matnni olish

# Traing new model
model.train()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(11):  # Epochs
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} completed")

# Save New AI Model!
model.save_pretrained('YourModelNameAndPath')
tokenizer.save_pretrained('./models/TestTokenizer')
