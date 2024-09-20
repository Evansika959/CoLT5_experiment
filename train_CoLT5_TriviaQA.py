from datasets import load_dataset

# Load the train split of TriviaQA
dataset = load_dataset('trivia_qa', 'unfiltered', split='train')

# Process the Dataset
from transformers import T5Tokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess_function(examples):
    inputs = [f"trivia question: {question}" for question in examples['question']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')

    # Handle answers
    answers = [answer['value'][0] if len(answer['value']) > 0 else "" for answer in examples['answer']]
    labels = tokenizer(answers, max_length=128, truncation=True, padding='max_length')
    
    model_inputs['labels'] = labels['input_ids']
    model_inputs['labels'] = [label if label is not None else 0 for label in model_inputs['labels']]  # Avoid empty values
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)



from colt5_attention.transformer_block import ConditionalRoutedTransformerBlock, ConditionalRoutedAttention, ConditionalRoutedFeedForward, ConditionalRoutedCrossAttention
from transformers import T5ForConditionalGeneration

# Load pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Replace default attention blocks with Conditional Routed ones
for layer in model.encoder.block:
    layer.layer[0] = ConditionalRoutedAttention(dim=512, num_heavy_tokens_q=32, num_heavy_tokens_kv=32)  # Conditional attention layer
    layer.layer[1] = ConditionalRoutedFeedForward(dim=512, num_heavy_tokens=32)  # Conditional feed-forward layer

for layer in model.decoder.block:
    layer.layer[0] = ConditionalRoutedAttention(dim=512,num_heavy_tokens_q=32, num_heavy_tokens_kv=32)  # Self-attention
    layer.layer[1] = ConditionalRoutedCrossAttention(dim=512,num_tokens_q=512, num_tokens_kv=512)  # Cross-attention
    layer.layer[2] = ConditionalRoutedFeedForward(dim=512,num_heavy_tokens=32)  # Feed-forward


model.to('cuda')

# Define the Training Loop

import torch
from torch.utils.data import DataLoader

# Use the DataCollatorWithPadding to pad inputs dynamically
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# Remove unnecessary columns after tokenization
tokenized_dataset = tokenized_dataset.remove_columns(['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer', 'attention_mask'])

# Print data types of all columns in the tokenized dataset
for column in tokenized_dataset.features:
    print(f"Column: {column}, Type: {tokenized_dataset.features[column]}")


train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()

from tqdm import tqdm  # Import tqdm

# for epoch in range(4):  # Adjust based on your needs
#     for batch in train_loader:
#         input_ids = batch['input_ids'].to('cuda')
# #        attention_mask = batch['attention_mask'].to('cuda')
#         labels = batch['labels'].to('cuda')

#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch + 1} finished with loss: {loss.item()}")

epochs = 4  # Adjust based on your needs
for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Update progress bar with loss
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

# Save the model
model.save_pretrained('./colt5_triviaqa_model')