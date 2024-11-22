import numpy as np
import argparse
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time

# Argument parser
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--number-of-clauses", default=1000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=3, type=int)
    parser.add_argument("--hypervector-size", default=64, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=64, type=int)
    parser.add_argument("--imdb_num_words", default=10000, type=int)
    parser.add_argument("--imdb_index_from", default=2, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

# Load dataset
print("Preparing dataset")
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)

word_to_id = imdb.get_word_index()
word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value: key for key, value in word_to_id.items()}

print("Preparing dataset.... Done!")

number_of_nodes = 500  # Increase sequence length

# Pad sequences for uniform length
train_x = pad_sequences(train_x, maxlen=number_of_nodes, padding='post', truncating='post')
test_x = pad_sequences(test_x, maxlen=number_of_nodes, padding='post', truncating='post')

# Create symbols
symbols = [id_to_word.get(i, f"UNK_{i}") for i in range(args.imdb_num_words)]
print("Symbols:", symbols[:10])

# Initialize training graphs
graphs_train = Graphs(
    len(train_x),
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits
)

print("Train data graph setup done.")
def configure_graphs(graphs, data_x, label="train"):
    for graph_id in range(len(data_x)):
        graphs.set_number_of_graph_nodes(graph_id, number_of_nodes)

    graphs.prepare_node_configuration()

    for graph_id in range(len(data_x)):
        for node_id in range(number_of_nodes):
            # Dynamically set the number of edges for each node
            if node_id == 0 or node_id == number_of_nodes - 1:
                num_edges = 2  # First and last nodes
            elif node_id == 1 or node_id == number_of_nodes - 2:
                num_edges = 3  # Second and second-to-last nodes
            else:
                num_edges = 4  # All intermediate nodes
            graphs.add_graph_node(graph_id, f"Node_{node_id}", num_edges)

    graphs.prepare_edge_configuration()

    for graph_id in range(len(data_x)):
        for node_id in range(number_of_nodes):
            # Add edges based on node position
            if node_id == 0:  # First node
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + 1}", "+1")
                if number_of_nodes > 2:
                    graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + 2}", "+2")
            elif node_id == number_of_nodes - 1:  # Last node
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - 1}", "-1")
                if number_of_nodes > 2:
                    graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - 2}", "-2")
            elif node_id == 1:  # Second node
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - 1}", "-1")
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + 1}", "+1")
                if number_of_nodes > 3:
                    graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + 2}", "+2")
            elif node_id == number_of_nodes - 2:  # Second-to-last node
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - 2}", "-2")
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - 1}", "-1")
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + 1}", "+1")
            else:  # Intermediate nodes
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - 2}", "-2")
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - 1}", "-1")
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + 1}", "+1")
                graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + 2}", "+2")

    for graph_id, review in enumerate(data_x):
        for node_id, token in enumerate(review):
            if token > 0:
                word = id_to_word.get(token, f"UNK_{token}")
                graphs.add_graph_node_property(graph_id, f"Node_{node_id}", word)

    graphs.encode()
    print(f"Graph construction completed for the {label} data.")


# Configure training graphs
configure_graphs(graphs_train, train_x, label="training")

# Initialize testing graphs
graphs_test = Graphs(
    len(test_x),
    init_with=graphs_train
)

print("Test data graph setup done.")

# Configure testing graphs
configure_graphs(graphs_test, test_x, label="testing")

# Initialize Tsetlin Machine
tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits=args.number_of_state_bits,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
    double_hashing=args.double_hashing,
    grid=(16*13*4,1,1),
    block=(128,1,1)
)

# Training and evaluation
for epoch in range(args.epochs):
    start_training = time()
    tm.fit(graphs_train, train_y, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    train_predictions = tm.predict(graphs_train)
    test_predictions = tm.predict(graphs_test)
    train_accuracy = 100 * (train_predictions == train_y).mean()
    test_accuracy = 100 * (test_predictions == test_y).mean()
    stop_testing = time()

    print(f"Epoch {epoch+1}/{args.epochs}")
    print(f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Training Time: {stop_training - start_training:.2f}s, Testing Time: {stop_testing - start_testing:.2f}s")
