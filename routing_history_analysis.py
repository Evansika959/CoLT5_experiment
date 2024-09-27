import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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

    # print("data length: ", num_data)
    
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
        # print("tier0: ", tier0)
        # print("tier1: ", tier1)
        # print("tier2: ", tier2)
        # print("tier3: ", tier3)
        # print("similarity: ", sum_similarity / len(selected_kv_batch))

        tier0_ratio.append(tier0/len(selected_kv_batch))
        tier1_ratio.append(tier1/len(selected_kv_batch))
        tier2_ratio.append(tier2/len(selected_kv_batch))
        tier3_ratio.append(tier3/len(selected_kv_batch))

    print("tier0_ratio: ", sum(tier0_ratio) / len(tier0_ratio))
    print("tier1_ratio: ", sum(tier1_ratio) / len(tier1_ratio))
    print("tier2_ratio: ", sum(tier2_ratio) / len(tier2_ratio))
    print("tier3_ratio: ", sum(tier3_ratio) / len(tier3_ratio))

    tier0_ratio_avg = sum(tier0_ratio) / len(tier0_ratio)
    tier1_ratio_avg = sum(tier1_ratio) / len(tier1_ratio)
    tier2_ratio_avg = sum(tier2_ratio) / len(tier2_ratio)
    tier3_ratio_avg = sum(tier3_ratio) / len(tier3_ratio)

    return similarity_scores

def plot_distribution_bar(tier0_ratio, tier1_ratio, tier2_ratio, tier3_ratio, layernum):
    plt.figure(figsize=(10, 6))
    
    # Generate the x-axis positions (layer numbers)
    layers = np.arange(layernum)
    
    # Plot stacked bars
    plt.bar(layers, tier0_ratio, label='Tier 0', color='skyblue', edgecolor='black', alpha=0.7)
    plt.bar(layers, tier1_ratio, bottom=tier0_ratio, label='Tier 1', color='lightgreen', edgecolor='black', alpha=0.7)
    plt.bar(layers, tier2_ratio, bottom=np.array(tier0_ratio) + np.array(tier1_ratio), label='Tier 2', color='lightcoral', edgecolor='black', alpha=0.7)
    plt.bar(layers, tier3_ratio, bottom=np.array(tier0_ratio) + np.array(tier1_ratio) + np.array(tier2_ratio), label='Tier 3', color='lightgoldenrodyellow', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('Tier Ratio')
    plt.title('Stacked Bar Chart of Tier Ratios per Layer')

    # Add x-ticks for layer numbers
    plt.xticks(layers)

    # Add legend
    plt.legend()

    # Save the plot
    plt.savefig("routing_analysis_stacked_bar.png")

def plot_similarity_histogram(similarity_scores):
    plt.figure(figsize=(10, 6))
    
    # Plot a histogram of similarity scores
    plt.bar(range(len(similarity_scores)), similarity_scores, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Layer Number')
    plt.ylabel('similarity')
    plt.title('Histogram of Similarity Scores')
    
    plt.savefig("routing_analysis_bar.png")

def plot_similarity_scores(layer_similarity):
    plt.figure(figsize=(10, 6))
    
    # Plot similarity scores for each layer
    for layer, scores in layer_similarity.items():
        plt.plot(scores, label=f'Layer {layer}')
    
    # Add labels and title
    plt.xlabel('DataLoad Number')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Scores Across Different Layers')
    plt.legend()  # Show the legend
    plt.savefig('routing_analysis.png')
    # plt.show()

def main():
    # Load the routing history
    router_histories = load_routing_history('routing_history.pkl')
    print("Routing histories have been loaded successfully.")
    
    # Define the number of encoder layers
    num_encoder_layers = 6  # Adjust based on your model's architecture
    
    # Initialize a dictionary to store similarity scores per layer
    layer_similarity = {}
    
    layer_scores = []

    # Compute similarity scores for each layer
    for layer in range(num_encoder_layers):
        print(f"\nProcessing Layer {layer}...")
        similarity_scores = compare_similarity_per_batch(layer, router_histories)
        layer_similarity[layer] = similarity_scores
        print(f"Completed Layer {layer}: {len(similarity_scores)} data_loads compared.")
        print(f"Average Similarity: {sum(similarity_scores) / len(similarity_scores):.4f}")
        layer_scores.append(sum(similarity_scores) / len(similarity_scores))
    
    plot_similarity_scores(layer_similarity)
    plot_similarity_histogram(layer_scores)
    print(layer_scores)
    
    # Optionally, save the similarity scores for further analysis
    with open('layer_similarity_scores.pkl', 'wb') as f:
        pickle.dump(layer_similarity, f)
    
    print("Similarity scores have been saved to 'layer_similarity_scores.pkl'.")

if __name__ == "__main__":
    main()
