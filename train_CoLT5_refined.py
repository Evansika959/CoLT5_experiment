import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorWithPadding
from colt5_attention.transformer_block import ConditionalRoutedTransformerBlock, ConditionalRoutedDecoderBlock
from colt5_attention.colt5_model import CoLT5
from tqdm import tqdm
import os
import matplotlib.pyplot as plt  # For plotting loss curves

# ============================
# 1. Data Loading and Preprocessing
# ============================

# Load the train split of TriviaQA
dataset = load_dataset('trivia_qa', 'unfiltered', split='train')

# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def preprocess_function(examples):
    # Prepare inputs
    inputs = [f"trivia question: {question}" for question in examples['question']]
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    # Prepare labels
    answers = [answer['value'] if len(answer['value']) > 0 else "" for answer in examples['answer']]
    labels = tokenizer(
        answers,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    model_inputs['labels'] = labels['input_ids']

    # Convert attention_mask to boolean type
    model_inputs['attention_mask'] = model_inputs['attention_mask'].bool()
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns
columns_to_remove = ['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer']
tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

# ============================
# 2. Dataset Splitting
# ============================

# Split the dataset into training and validation sets (90% train, 10% validation)
train_size = int(0.9 * len(tokenized_dataset))
val_size = len(tokenized_dataset) - train_size
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])

# Initialize DataCollator
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

# ============================
# 3. Model Initialization
# ============================

# Load CoLT5 model
model = CoLT5(num_layers=6, dim=512).to('cuda')

# ============================
# 4. Training Setup
# ============================

# Define Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Define Loss Function
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Directory for checkpoints
checkpoint_dir = './checkpoints_925'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Initialize lists to track loss
train_losses = []
val_losses = []

# Early Stopping Parameters
patience = 3
best_val_loss = float('inf')
counter = 0

# ============================
# 5. Training Loop with Validation and Loss Tracking
# ============================

epochs = 10  # Increased number of epochs for better convergence

for epoch in range(epochs):
    # Training Phase
    model.train()
    epoch_train_loss = 0
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        input_ids = batch['input_ids'].to('cuda')
        mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        # Initialize decoder_input_ids with pad tokens and shift labels
        decoder_input_ids = torch.full((labels.size(0), labels.size(1)), tokenizer.pad_token_id, dtype=torch.long).to('cuda')
        decoder_input_ids[:, 1:] = labels[:, :-1]  # Shift labels for decoder input
        decoder_input_ids[:, 0] = tokenizer.pad_token_id  # Ensure the first token is pad

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mask=mask)

        # Compute loss
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backward pass
        loss.backward()

        # Gradient Clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        epoch_train_loss += loss.item()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    # Compute average training loss for the epoch
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Phase
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to('cuda')
            mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            # Initialize decoder_input_ids with pad tokens and shift labels
            decoder_input_ids = torch.full((labels.size(0), labels.size(1)), tokenizer.pad_token_id, dtype=torch.long).to('cuda')
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = tokenizer.pad_token_id

            # Forward pass
            logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mask=mask)

            # Compute loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            epoch_val_loss += loss.item()

    # Compute average validation loss for the epoch
    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Scheduler step based on validation loss
    scheduler.step(avg_val_loss)

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        # Save the best model
        torch.save(model.state_dict(), f'{checkpoint_dir}/best_colt5.pth')
        print(f"Epoch {epoch + 1} improved. Saving best model.")
    else:
        counter += 1
        print(f"Epoch {epoch + 1} did not improve.")
        if counter >= patience:
            print("Early stopping triggered.")
            break

    # Save a checkpoint at the end of each epoch
    torch.save(model.state_dict(), f'{checkpoint_dir}/colt5_epoch_{epoch+1}.pth')
    print(f"Epoch {epoch + 1} completed. Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# ============================
# 6. Save the Final Model
# ============================

# Save the final model
model.save_pretrained('./colt5_triviaqa_model')

# ============================
# 7. Plotting the Loss Curves
# ============================

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')  # Save the plot as an image file
plt.show()
