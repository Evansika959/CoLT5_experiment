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

# Function to generate answer
def generate_answer(model, input_ids, max_length=128):
    # Create an empty tensor for the decoder input
    decoder_input_ids = torch.full((1, 1), tokenizer.pad_token_id, dtype=torch.long).to('cuda')  # Start with a padding token

    print(decoder_input_ids.shape)

    # Generate up to max_length tokens
    for _ in range(max_length):
        # Forward pass
        outputs = model(input_ids=input_ids['input_ids'], decoder_input_ids=decoder_input_ids, mask=attention_mask)
        logits = outputs  # Assuming the last output is logits

        # Get the predicted token (argmax)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        
        # Append the predicted token to the decoder input
        decoder_input_ids = torch.cat((decoder_input_ids, torch.tensor([[next_token_id]], device='cuda')), dim=1)

        # If the predicted token is the end of sequence token, break
        if next_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

# Generate answer
predicted_answer = generate_answer(model, input_ids)

# Print the results
print(f"Sample Question: {sample_question}")
print(f"Expected Answer: {labels}")
print(f"Predicted Answer: {predicted_answer}")
