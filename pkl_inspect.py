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
        

if __name__ == "__main__":
    routing_history = load_routing_history('routing_history.pkl')
    inspect_routing_history(routing_history)
