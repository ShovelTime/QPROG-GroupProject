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
    #Make 2-star connection with bridge
    edges_8 = [
        (0,1), (0,2), (0,3), 
        (4,5), (4,6), (4,7), 
        (0,4)                
    ]
    for u, v in edges_8:
        G_8.add_edge(u, v)

    G_16 = Graph(16)
    #4 star connection with bridges
    centers = [0, 4, 8, 12]
    for c in centers:
        for leaf in range(c+1, c+4):
            G_16.add_edge(c, leaf)
    G_16.add_edge(0, 4)
    G_16.add_edge(4, 8)
    G_16.add_edge(8, 12)

    return [G_4, G_8, G_16]

def test_adj_circuit():
    graphs = get_test_graphs()
    
    G = graphs[0]

    print("########################### Adjacent circuit Test ###########################")

    true_edges = [2,3]
    true_edges_bits = numbers_to_bit_matrix(np.array(true_edges, np.uint32), G.get_bit_count())
    false_edges = [0,3]
    false_edges_bits = numbers_to_bit_matrix(np.array(false_edges, np.uint32), G.get_bit_count())

    circuit = QuantumCircuit(5,1)
    counts = Adj(G, circuit, true_edges_bits[0], true_edges_bits[1], 4)

    if bool(int(list(counts.keys())[0])) != G.is_connected(2,3):
        print(circuit)
        raise Exception("Test key mismatch! circuit output:", str(counts.keys()), "classical output: ", counts)
    else:
        print("G_4 is_true passed")

    counts = Adj(G, circuit, false_edges_bits[0] ,false_edges_bits[1], 4)

    if int(list(counts.keys())[0]) != G.is_connected(0,3):
        print(circuit)
        raise Exception("Test key mismatch! circuit output:", str(counts.keys()), "classical output: ", counts)
    else:
        print("G_4 is_false passed")

        
    
    print("########################### DONE ###########################")

def test_dominated_vertex():

    graphs = get_test_graphs()
    G = graphs[0]
    bit_count = G.get_bit_count()
    print("########################### Dominated Vertex Test ###########################")
    candidate_u = 0
    u_bits = np.array([0,0], dtype=np.uint32)
    valid_candidate_set = np.array([0,1,3], dtype=np.uint32)
    invalid_candidate_set = np.array([0,1,2], dtype=np.uint32)
    valid_result = classical_dominating_node(0, G, valid_candidate_set)
    invalid_result = classical_dominating_node(0,G,invalid_candidate_set)

    assert valid_result
    assert not invalid_result

    valid_bits = numbers_to_bit_matrix(valid_candidate_set, bit_count)
    invalid_bits = numbers_to_bit_matrix(invalid_candidate_set, bit_count)

    total_qbit_count = (bit_count * (len(valid_candidate_set) + 1)) + 3

    circuit = QuantumCircuit(total_qbit_count, 1)
    

    counts = Dominated(G, circuit, valid_bits, u_bits, auxiliary=np.arange(total_qbit_count - 3, total_qbit_count - 1, dtype=np.uint32), output=total_qbit_count - 1)
    if bool(int(list(counts.keys())[0])) != valid_result:
        print(circuit)
        raise Exception("Test key mismatch! circuit output:", str(counts.keys()), "classical output: ", valid_result)
    else:
        print("G_4 is_true passed")

    counts = Dominated(G, circuit, invalid_bits, u_bits, auxiliary=np.arange(total_qbit_count - 3, total_qbit_count - 1, dtype=np.uint32), output=total_qbit_count - 1)

    if bool(int(list(counts.keys())[0])) != invalid_result:
        print(circuit)
        raise Exception("Test key mismatch! circuit output:", str(counts.keys()), "classical output: ", invalid_result)
    else:
        print("G_4 is_false passed")

    print("########################### DONE ###########################")

def test_dominated_set():

    graphs = get_test_graphs()
    G = graphs[0]
    bit_count = G.get_bit_count()
    print("########################### Dominated Vertex Test ###########################")
    candidate_u = 0
    u_bits = np.array([0,0], dtype=np.uint32)
    valid_candidate_set = np.array([0,1,3], dtype=np.uint32)
    invalid_candidate_set = np.array([0,1,2], dtype=np.uint32)
    valid_result = classical_dominating_node(0, G, valid_candidate_set)
    invalid_result = classical_dominating_node(0,G,invalid_candidate_set)

    assert valid_result
    assert not invalid_result

    valid_bits = numbers_to_bit_matrix(valid_candidate_set, bit_count)
    invalid_bits = numbers_to_bit_matrix(invalid_candidate_set, bit_count)

    total_qbit_count = (bit_count * (len(valid_candidate_set) + 1)) + 3

    circuit = QuantumCircuit(total_qbit_count, 1)
    

    counts = Dominated(G, circuit, valid_bits, u_bits, auxiliary=np.arange(total_qbit_count - 3, total_qbit_count - 1, dtype=np.uint32), output=total_qbit_count - 1)
    if bool(int(list(counts.keys())[0])) != valid_result:
        print(circuit)
        raise Exception("Test key mismatch! circuit output:", str(counts.keys()), "classical output: ", valid_result)
    else:
        print("G_4 is_true passed")

    counts = Dominated(G, circuit, invalid_bits, u_bits, auxiliary=np.arange(total_qbit_count - 3, total_qbit_count - 1, dtype=np.uint32), output=total_qbit_count - 1)

    if bool(int(list(counts.keys())[0])) != invalid_result:
        print(circuit)
        raise Exception("Test key mismatch! circuit output:", str(counts.keys()), "classical output: ", invalid_result)
    else:
        print("G_4 is_false passed")

    print("########################### DONE ###########################")


def main():
    test_adj_circuit()

    test_dominated_vertex()



if __name__=="__main__":
    main()