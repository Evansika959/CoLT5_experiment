import torch
import random
from datasets import load_dataset
from transformers import T5Tokenizer
from colt5_attention.colt5_model import CoLT5
from colt5_attention.transformer_block import CoordinateDescentRouter

import torch
from collections import defaultdict

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
model = CoLT5(num_layers=6, dim=512).to('cuda')
model.load_state_dict(torch.load('./checkpoints_925/best_colt5.pth'))
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

# Print the sample question and expected answer
print(f"Sample Question: {sample_question}")
# print(f"Question Tokenized: {input_ids}")
print(f"Expected Answer: {labels}")
# print(f"Answer Tokenized: {labels_tokens}")

# Generate the answer
predicted_answer = model.generate(input_ids=input_ids['input_ids'], encoder_mask=attention_mask, max_new_tokens=10, temperature=1.0, top_k=None, keep_routing_history=True, verbose=True)

generated_ids = predicted_answer[0].cpu().tolist()
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Extract routing histories
router_histories = extract_router_history(model)

# Display the routing histories
for router_name, history in router_histories.items():
    print(f"Router: {router_name}")
    print(f"Selected Indices: {history['selected_indices']}")
    # print(f"Selected Scores: {history['selected_scores']}")
    # print(f"mask: {history['input_mask']}")

# Print the results
print(f"Predicted Answer: {generated_text}")
print(f"Predicted Answer Tokenized: {generated_ids}")
print(f"mask: {attention_mask}")
print(f"Generated Tokens: {generated_ids}")

# Assuming 'model' is your CoLT5 model instance and routing history has been kept during training/inference



