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
    kv_history = router_histories[kv_router_name]['selected_indices']
    ffn_history = router_histories[ffn_router_name]['selected_indices']
    
    if len(kv_history) != len(ffn_history):
        print(f"Warning: Mismatch in number of batches for layer {layer_num}.")
    
    num_data = min(len(kv_history), len(ffn_history))

    print("data length: ", num_data)
    
    similarity_scores = []
    
    tier0_ratio = []
    tier1_ratio = []
    tier2_ratio = []
    tier3_ratio = []

    for data_idx in tqdm(range(num_data), desc=f"Comparing Layer {layer_num} Batches"):
        # Each history entry is a list of selected indices per sample in the batch
        # e.g., [[1, 2, 3, 4], [5,6,7,8], ...] for batch_size samples
        
        selected_kv_batch = kv_history[data_idx]  # List of lists
        selected_ffn_batch = ffn_history[data_idx]  # List of lists
        
        print("batch length: ", len(selected_kv_batch))

        sum_similarity = 0
        tier0 = 0
        tier1 = 0
        tier2 = 0
        tier3 = 0

        for batch_idx in range(len(selected_kv_batch)):
            selected_kv = selected_kv_batch[batch_idx]
            selected_ffn = selected_ffn_batch[batch_idx]

            # Convert tensors to Python sets for set operations
            selected_kv_set = set(selected_kv.tolist())
            selected_ffn_set = set(selected_ffn.tolist())

            if data_idx == 1 and batch_idx == 1:
                print(selected_kv)
                print(selected_ffn)

            intersection = selected_kv_set.intersection(selected_ffn_set)
        
            similarity = len(intersection) / len(selected_kv_set)

            sum_similarity += similarity

            if similarity == 0.25:
                tier3 += 1

            if similarity == 0.5:
                tier2 += 1

            if similarity == 0.75:
                tier1 += 1

            if similarity == 1:
                tier0 += 1
        
        similarity_scores.append(sum_similarity / len(selected_kv_batch))
        print("tier0: ", tier0)
        print("tier1: ", tier1)
        print("tier2: ", tier2)
        print("tier3: ", tier3)
        print("similarity: ", sum_similarity / len(selected_kv_batch))

        tier0_ratio.append(tier0/len(selected_kv_batch))
        tier1_ratio.append(tier1/len(selected_kv_batch))
        tier2_ratio.append(tier2/len(selected_kv_batch))
        tier3_ratio.append(tier3/len(selected_kv_batch))

    print("tier0_ratio: ", sum(tier0_ratio) / len(tier0_ratio))
    print("tier1_ratio: ", sum(tier1_ratio) / len(tier1_ratio))
    print("tier2_ratio: ", sum(tier2_ratio) / len(tier2_ratio))
    print("tier3_ratio: ", sum(tier3_ratio) / len(tier3_ratio))

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
    plt.savefig('routing_analysis.png')  # Save the plot as an image file
    # plt.show()

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
        print(f"Completed Layer {layer}: {len(similarity_scores)} data_loads compared.")
        print(f"Average Similarity: {sum(similarity_scores) / len(similarity_scores):.4f}")
    
    # Example: Plot histogram for each layer
    for layer, scores in layer_similarity.items():
        plot_similarity_histogram(scores, layer)
    
    # Optionally, save the similarity scores for further analysis
    with open('layer_similarity_scores.pkl', 'wb') as f:
        pickle.dump(layer_similarity, f)
    
    print("Similarity scores have been saved to 'layer_similarity_scores.pkl'.")

if __name__ == "__main__":
    main()
