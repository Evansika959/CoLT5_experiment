import torch
import random
from datasets import load_dataset
from transformers import T5Tokenizer
from colt5_attention.colt5_model import CoLT5

# Load the model and tokenizer
model = CoLT5(num_layers=6, dim=512).to('cuda')
model.load_state_dict(torch.load('./checkpoints/colt5_epoch_1.pth'))
model.eval()  # Set the model to evaluation mode

tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the TriviaQA dataset
dataset = load_dataset('trivia_qa', 'unfiltered', split='train')

# Randomly select a sample from the dataset
sample_index = random.randint(0, len(dataset) - 1)
sample = dataset[sample_index]

# Prepare the input question
sample_question = sample['question']
input_text = f"trivia question: {sample_question}"

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to('cuda')
attention_mask = input_ids['attention_mask'].to('cuda')

# Prepare the labels
labels = sample['answer'][0]['value'][0] if sample['answer'] else ""
labels_tokens = tokenizer(labels, return_tensors='pt', padding='max_length', truncation=True, max_length=128).input_ids.to('cuda')

# Perform inference
with torch.no_grad():
    logits = model(input_ids=input_ids['input_ids'], decoder_input_ids=labels_tokens, mask=attention_mask)

# Get the predicted token IDs
predicted_ids = torch.argmax(logits, dim=-1)

# Decode the predicted token IDs to text
predicted_answer = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

# Print the results
print(f"Sample Question: {sample_question}")
print(f"Expected Answer: {labels}")
print(f"Predicted Answer: {predicted_answer}")
