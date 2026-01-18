from dominatingset import *


#############################################################
########################### TESTS ###########################
#############################################################

def get_test_graphs() -> list[Graph]:
    G_4 = Graph(4)
    G_4.add_edge(0, 1)
    G_4.add_edge(0, 3)
    G_4.add_edge(1, 2)
    G_4.add_edge(1, 3)
    G_4.add_edge(2, 3)
    G_4.print()

    G_8 = Graph(8)

    G_16 = Graph(16)

    return [G_4, G_8, G_16]

def test_adj_circuit():
    graphs = get_test_graphs()
    
    G = graphs[0]

    print("########################### Adjacent circuit Test ###########################")

    edges = numbers_to_bit_matrix(np.array([2,3], np.uint32), G.get_bit_count())
    circuit = QuantumCircuit(5,1)
    counts = init_and_run_adjacent_circuit(G, circuit, edges[0], edges[1], 4)

    if '1' not in counts.keys() or not G.is_connected(2,3):
        print(circuit)
        raise Exception("Test key mismatch! circuit output:", str(counts.keys()), "classical output: ", counts)
    else:
        print("G_4 passed")
    
    print("########################### DONE ###########################")