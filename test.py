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



def main():
    test_adj_circuit()



if __name__=="__main__":
    main()