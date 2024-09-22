import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

# Load the train split of TriviaQA
dataset = load_dataset('trivia_qa', 'unfiltered', split='train')

# Process the Dataset
from transformers import T5Tokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess_function(examples):
    inputs = [f"trivia question: {question}" for question in examples['question']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length',return_tensors='pt')

    # Handle answers
    answers = [answer['value'][0] if len(answer['value']) > 0 else "" for answer in examples['answer']]
    labels = tokenizer(answers, max_length=128, truncation=True, padding='max_length',return_tensors='pt')
    
    model_inputs['labels'] = labels['input_ids']
    model_inputs['labels'] = [label if label is not None else 0 for label in model_inputs['labels']]  # Avoid empty values

    # Convert attention_mask to boolean type
    model_inputs['attention_mask'] = model_inputs['attention_mask'].bool()
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

def is_sequence_too_short(input_ids, min_length=10):
    # Check if the sequence is shorter than the minimum length
    return input_ids.size(1) < min_length


from colt5_attention.transformer_block import ConditionalRoutedTransformerBlock, ConditionalRoutedDecoderBlock

# Define the CoLT5 Encoder

seq_len = 512
num_heavy_tokens = 32

class CoLT5Encoder(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5Encoder, self).__init__()
        self.layers = nn.ModuleList([ConditionalRoutedTransformerBlock(dim, num_heavy_attn_tokens_kv=num_heavy_tokens, num_heavy_attn_tokens_q=num_heavy_tokens, num_heavy_ff_tokens=num_heavy_tokens) for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(32128, dim)  # Vocab size of 32,128 tokens

    def forward(self, input_ids, mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Define the CoLT5 Decoder with Cross-Attention
class CoLT5Decoder(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5Decoder, self).__init__()
        self.layers = nn.ModuleList([ConditionalRoutedDecoderBlock(dim, num_heavy_attn_tokens_kv=num_heavy_tokens, num_heavy_attn_tokens_q=num_heavy_tokens, num_heavy_ff_tokens=num_heavy_tokens) for _ in range(num_layers)])
        self.embed_tokens = nn.Embedding(32128, dim)  # Vocab size of 32,128 tokens

    def forward(self, input_ids, encoder_hidden_states, mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, encoder_hidden_states, mask)
        return x


# Define the full CoLT5 model
class CoLT5(nn.Module):
    def __init__(self, num_layers, dim):
        super(CoLT5, self).__init__()
        self.encoder = CoLT5Encoder(num_layers=num_layers, dim=dim)
        self.decoder = CoLT5Decoder(num_layers=num_layers, dim=dim)
        self.lm_head = nn.Linear(dim, 32128)  # Output layer to vocab size

    def forward(self, input_ids, decoder_input_ids, mask=None):
        # Encode the input
        encoder_hidden_states = self.encoder(input_ids, mask)
        # Decode the input
        decoder_hidden_states = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_states, mask=mask)
        # Generate final token predictions
        logits = self.lm_head(decoder_hidden_states)
        return logits

# Load CoLT5 model
model = CoLT5(num_layers=6, dim=512).to('cuda')

# Define the Training Loop

import torch
from torch.utils.data import DataLoader

# Use the DataCollatorWithPadding to pad inputs dynamically
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# Remove unnecessary columns after tokenization
tokenized_dataset = tokenized_dataset.remove_columns(['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer'])

# Print data types of all columns in the tokenized dataset
for column in tokenized_dataset.features:
    print(f"Column: {column}, Type: {tokenized_dataset.features[column]}")


train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()

from tqdm import tqdm  # Import tqdm

epochs = 4  # Adjust based on your needs
for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids = batch['input_ids'].to('cuda')
        mask = batch['attention_mask'].to('cuda')

        labels = batch['labels'].to('cuda')
        # labels = batch['labels'].squeeze().to('cuda')

        decoder_input_ids = labels.clone()  # Shift labels for decoder input

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mask=mask)
        
        # Loss function: Cross-Entropy
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())

    # Save a checkpoint at the end of each epoch
    torch.save(model.state_dict(), f'./checkpoints/colt5_epoch_{epoch+1}.pth')
    print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

# Save the model
model.save_pretrained('./colt5_triviaqa_model')