import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer
from tqdm import tqdm  # Import tqdm for the progress bar
from colt5_attention.colt5_model import CoLT5
import torch.nn as nn

# Load the model and tokenizer
model = CoLT5(num_layers=6, dim=512).to('cuda')  # Adjust this if your architecture changes
model.load_state_dict(torch.load('./checkpoints/colt5_epoch_1.pth'))
model.eval()  # Set the model to evaluation mode

tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load and preprocess the test dataset
test_dataset = load_dataset('trivia_qa', 'unfiltered', split='validation')  # Or 'test' if available

def preprocess_function(examples):
    inputs = [f"trivia question: {question}" for question in examples['question']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='pt')

    # Handle answers
    answers = [answer['value'][0] if len(answer['value']) > 0 else "" for answer in examples['answer']]
    labels = tokenizer(answers, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize the test dataset
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# DataLoader for the test set
test_loader = DataLoader(tokenized_test_dataset, batch_size=8)

# Evaluate the model
total_loss = 0
with torch.no_grad():  # Disable gradient calculation for evaluation
    loop = tqdm(test_loader, leave=True, desc="Evaluating")
    for batch in loop:
        input_ids = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')

        # Forward pass
        logits = model(input_ids=input_ids, decoder_input_ids=labels)
        
        # Loss function: Cross-Entropy
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())  # Update progress bar with the current loss

average_loss = total_loss / len(test_loader)
print(f"Average Test Loss: {average_loss}")
