import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorWithPadding
from tqdm import tqdm  # Import tqdm for the progress bar
from colt5_attention.colt5_model import CoLT5
from colt5_attention.transformer_block import CoordinateDescentRouter
import torch.nn as nn

def extract_router_history(model):
    """
    Extracts the routing history from all CoordinateDescentRouter instances within the model.

    Args:
        model (nn.Module): The CoLT5 model instance.

    Returns:
        dict: A dictionary where keys are router names (as per model's module hierarchy)
              and values are their corresponding routing histories.
    """
    router_histories = {}
    for name, module in model.named_modules():
        if isinstance(module, CoordinateDescentRouter):
            router_histories[name] = module.routing_history
    return router_histories

# Load the model and tokenizer
model = CoLT5(num_layers=6, dim=512).to('cuda')  # Adjust this if your architecture changes
model.load_state_dict(torch.load('./checkpoints_925/best_colt5.pth'))
model.eval()  # Set the model to evaluation mode

tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load and preprocess the test dataset
test_dataset = load_dataset('trivia_qa', 'unfiltered', split='validation')  # Or 'test' if available

def preprocess_function(examples):
    inputs = [f"trivia question: {question}" for question in examples['question']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length', return_tensors='pt')

    # Handle answers
    answers = [answer['value'] if len(answer['value']) > 0 else "" for answer in examples['answer']]
    labels = tokenizer(answers, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    
    model_inputs['labels'] = labels['input_ids']

    # Convert attention_mask to boolean type
    model_inputs['attention_mask'] = model_inputs['attention_mask'].bool()
    return model_inputs

# Tokenize the test dataset
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns after tokenization
tokenized_dataset = tokenized_test_dataset.remove_columns(['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer'])

# Print data types of all columns in the tokenized dataset
for column in tokenized_dataset.features:
    print(f"Column: {column}, Type: {tokenized_dataset.features[column]}")

# Use the DataCollatorWithPadding to pad inputs dynamically
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# DataLoader for the test set
test_loader = DataLoader(tokenized_dataset, batch_size=64, shuffle=True, collate_fn=data_collator)

# Evaluate the model
total_loss = 0
with torch.no_grad():  # Disable gradient calculation for evaluation
    loop = tqdm(test_loader, leave=True, desc="Evaluating")
    for batch in loop:
        input_ids = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')
        mask = batch['attention_mask'].to('cuda')

        decoder_input_ids = torch.full(labels.shape, tokenizer.pad_token_id, dtype=torch.long).to('cuda')
        decoder_input_ids[:,1:] = labels[:,:-1]  # Shift labels for decoder input

        # Forward pass
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mask=mask, keep_routing_history=True)
        
        # Loss function: Cross-Entropy
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())  # Update progress bar with the current loss

average_loss = total_loss / len(test_loader)
print(f"Average Test Loss: {average_loss}")

# Extract routing histories
router_histories = extract_router_history(model)

for router_name, history in router_histories.items():
    print(f"Router: {router_name}")
    print(f"Selected Indices: {history['selected_indices']}")
