import torch
import random
from datasets import load_dataset
from transformers import T5Tokenizer
from colt5_attention.colt5_model import CoLT5

# Load the model and tokenizer
model = CoLT5(num_layers=6, dim=512).to('cuda')
model.load_state_dict(torch.load('./checkpoint1.pth'))
model.eval()  # Set the model to evaluation mode

tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the TriviaQA dataset
dataset = load_dataset('trivia_qa', 'unfiltered', split='validation')

# Randomly select a sample until we find one with a non-empty answer
while True:
    index = random.randint(0, len(dataset) - 1)
    sample = dataset[index]
    
    if sample['answer']:  # Check if the answer is not empty
        break

# Prepare the input question
sample_question = sample['question']
input_text = f"trivia question: {sample_question}"

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).to('cuda')
attention_mask = input_ids['attention_mask'].bool().to('cuda')

# Prepare the labels
labels = sample['answer']['value'] if sample['answer'] else ""
labels_tokens = tokenizer(labels, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids.to('cuda')

# Generate the answer
predicted_answer = model.generate(input_ids=input_ids['input_ids'], encoder_mask=attention_mask, max_new_tokens=512, temperature=1.0, top_k=None)

# Print the results
print(f"Sample Question: {sample_question}")
print(f"Expected Answer: {labels}")
print(f"Predicted Answer: {predicted_answer}")
