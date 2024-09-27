import pickle

def load_routing_history(file_path='routing_history.pkl'):
    with open(file_path, 'rb') as f:
        routing_history = pickle.load(f)
    return routing_history

def inspect_routing_history(routing_history):
    for router_name, history in routing_history.items():
        print(f"Router: {router_name}")
        print(f"Type of history: {type(history)}")
        print(f"Number of batches: {len(history)}")
        if len(history) > 0:
            print(f"First batch: {history[0]}")
            if isinstance(history[0], list):
                print(f"Number of samples in first batch: {len(history[0])}")
                if len(history[0]) > 0:
                    print(f"First sample selected indices: {history[0][0]}")
        print("-" * 50)

if __name__ == "__main__":
    routing_history = load_routing_history('routing_history.pkl')
    inspect_routing_history(routing_history)
