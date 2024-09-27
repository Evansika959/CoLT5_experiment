import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_routing_history(file_path='routing_history.pkl'):
    """
    Loads the routing history from a pickle file.

    Args:
        file_path (str): Path to the pickle file containing routing histories.

    Returns:
        dict: Dictionary with router names as keys and routing histories as values.
    """
    with open(file_path, 'rb') as f:
        routing_history = pickle.load(f)
    return routing_history

def compare_similarity_per_batch(layer_num, router_histories):
    """
    Compares the similarity between ff.router and kv.router for each batch in a specific encoder layer.

    Args:
        layer_num (int): The encoder layer number (e.g., 0, 1, 2, ...).
        router_histories (dict): Dictionary containing routing histories.

    Returns:
        list: A list of similarity scores for each batch.
    """
    # Define router suffixes
    kv_router_suffix = ".conditional_attn.kv_router"
    ffn_router_suffix = ".conditional_ff.router"
    
    # Construct full router names
    kv_router_name = f'encoder.layers.{layer_num}{kv_router_suffix}'
    ffn_router_name = f'encoder.layers.{layer_num}{ffn_router_suffix}'
    
    # Retrieve routing histories
    kv_history = router_histories.get(kv_router_name, [])
    ffn_history = router_histories.get(ffn_router_name, [])
    
    if len(kv_history) != len(ffn_history):
        print(f"Warning: Mismatch in number of batches for layer {layer_num}.")
    
    num_batches = min(len(kv_history), len(ffn_history))
    
    similarity_scores = []
    
    for batch_idx in tqdm(range(num_batches), desc=f"Comparing Layer {layer_num} Batches"):
        # Each history entry is a list of selected indices per sample in the batch
        # e.g., [[1, 2, 3, 4], [5,6,7,8], ...] for batch_size samples
        
        selected_kv_batch = kv_history[batch_idx]  # List of lists
        selected_ffn_batch = ffn_history[batch_idx]  # List of lists
        
        # Flatten the lists to get all selected indices in the batch
        selected_kv = set([idx for sample in selected_kv_batch for idx in sample])
        selected_ffn = set([idx for sample in selected_ffn_batch for idx in sample])
        
        # Compute Jaccard Similarity
        intersection = selected_kv.intersection(selected_ffn)
        union = selected_kv.union(selected_ffn)
        
        if not union:
            similarity = 0.0
        else:
            similarity = len(intersection) / len(union)
        
        similarity_scores.append(similarity)
    
    return similarity_scores

def plot_similarity_histogram(similarity_scores, layer_num):
    """
    Plots a histogram of similarity scores for a specific layer.

    Args:
        similarity_scores (list): List of similarity scores.
        layer_num (int): The encoder layer number.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(similarity_scores, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Number of Batches')
    plt.title(f'Similarity Distribution between KV and FFN Routers in Layer {layer_num}')
    plt.grid(True)
    plt.show()

def main():
    # Load the routing history
    router_histories = load_routing_history('routing_history.pkl')
    print("Routing histories have been loaded successfully.")
    
    # Define the number of encoder layers
    num_encoder_layers = 6  # Adjust based on your model's architecture
    
    # Initialize a dictionary to store similarity scores per layer
    layer_similarity = {}
    
    # Compute similarity scores for each layer
    for layer in range(num_encoder_layers):
        print(f"\nProcessing Layer {layer}...")
        similarity_scores = compare_similarity_per_batch(layer, router_histories)
        layer_similarity[layer] = similarity_scores
        print(f"Completed Layer {layer}: {len(similarity_scores)} batches compared.")
    
    # Example: Plot histogram for each layer
    for layer, scores in layer_similarity.items():
        plot_similarity_histogram(scores, layer)
    
    # Optionally, save the similarity scores for further analysis
    with open('layer_similarity_scores.pkl', 'wb') as f:
        pickle.dump(layer_similarity, f)
    
    print("Similarity scores have been saved to 'layer_similarity_scores.pkl'.")

if __name__ == "__main__":
    main()
