import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model va Tokenizer yuklash
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# Ma'lumotlarni yuklash
with open('/Users/abdulhodiy/Desktop/alls/MyAIApp/my/Chat/chat-data/Data.txt', 'r') as file:
    texts = file.readlines()  # Matnni olish

# Modelni fine-tuning qilish
model.train()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(11):  # Epochlar soni
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} completed")

# Modelni saqlash
model.save_pretrained('/Users/abdulhodiy/Desktop/alls/MyAIApp/my/Chat/Models/AIcode/AICode-v1')
