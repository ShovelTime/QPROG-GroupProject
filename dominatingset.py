from typing import Any, Tuple
import numpy as np
import numpy.ma as ma
import numpy.typing as npt

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.circuit.library import Initialize
from qiskit.circuit import Instruction
from qiskit.circuit.library import MCXGate

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.providers.basic_provider import BasicSimulator


 # bit masks for converting 32-bit integers to binary matrices, see numbers_to_bit_matrix
__mask = np.array([1 << i for i in range(0,32)]) # bit masks are created in the natural order to ensure little endianess
#print(__mask)

class Graph:
    def __init__(self, n: int):
        '''
        Creates Graph stub, preallocates n sublists each holding n-1 elements,
        which allows for every vertices to be neighbors with every other.
        Empty neighbour slots are represented as the sentinel value -1,
        as vertices cannot be represented with a negative value.
        
        :param n: vertex count
        :type n: int
        '''
        self.n = n
        self.adj_list = np.full((n, n + 1), -1, np.int32, order='F')
        self.adj_list[:, -1] = 0

    def set_number_vertices(self, n: int):
        '''
        Updates vertex count within the graph.
        
        :param self: Description
        :param n: Description
        :type n: int
        '''
        if n == self.n: return

        old_n = self.n
        old_lens = np.array(self.adj_list[:, -1]).copy()

        if n > self.n :
            diff = n - self.n
            self.adj_list = np.pad(self.adj_list[:, :-1], ((0, diff), (0, diff + 1)), "constant", constant_values=-1)

            new_lens = np.zeros(shape=n, dtype=np.int32, order='F')
            new_lens[:old_n, -1] = old_lens
            self.adj_list = np.hstack((self.adj_list, new_lens))
        else:
            self.adj_list = np.resize(self.adj_list, (n, n + 1))
            self.adj_list[:, -1] = old_lens[:n]
    
    def add_edge(self, u: int, v: int):
        u_list = self.adj_list[u, :-1]
        u_len = self.adj_list[u,-1]

        if u_len >= u_list.shape[0]: 
            raise Exception(f"Vertex {u} has no remaining slots")
        
        if u_len == 0:
            u_list[0] = v
            self.adj_list[u,-1] = u_len + 1
        else:
            u_neighbours = u_list[:u_len]

            u_idx = np.searchsorted(u_neighbours, v)

            if u_idx < u_len and u_neighbours[u_idx] == v: return

            u_list[u_idx + 1 : u_len + 1] = u_list[u_idx: u_len]
            u_list[u_idx] = v
            self.adj_list[u,-1] = u_len + 1

        if u == v: return
        
        v_list = self.adj_list[v, :-1]
        v_len = self.adj_list[v,-1]
        if v_len >= v_list.shape[0]: 
            raise Exception(f"Vertex {u} has no remaining slots") 
        if v_len == 0:
            v_list[0] = u
            self.adj_list[v,-1] = v_len + 1
            return
        
        v_neighbours = v_list[:v_len]
        v_idx = np.searchsorted(v_neighbours, v)

        v_list[v_idx + 1 : v_len + 1] = v_list[v_idx: v_len]
        v_list[v_idx] = u
        self.adj_list[v,-1] = v_len + 1

    def print(self, with_mask: bool = True):
        print("Adjacency graph with ", self.n, "vertices.\n")

        if with_mask:
            mask = -1
        else:
            mask = -2_147_483_648 # Dummy value that will filter nothing
        adj_filtered: ma.MaskedArray = ma.masked_equal(self.adj_list[:, :-1], mask)
        for i in range(0, self.n):
            adjacents = adj_filtered[i, :]
            adjacents = adjacents[~adjacents.mask] # Removed masked values so that they are not printed
            if len(adjacents) == 0:
                adjacents = "Empty"
            print("Vertex", i, ": ", adjacents)
    
    def clear(self):
        self.adj_list = np.full((self.n, self.n), -1, np.int32, order='F')
        self.adj_list[:,-1] = 1


    def read_from_file(self, file_path: str):
        with open(file_path) as f:
            lines = f.readlines()
            n = int(lines[0])
            self.set_number_vertices(n)
            self.clear()
            for line in lines:
                split = line.strip().split(' ')
                u, v = int(split[0]), int(split[1])
                self.add_edge(u, v)

    def is_connected(self, u: int, v: int) -> bool:
        edges = self.adj_list[u]
        edges = edges[:edges[-1]]
        idx =  np.searchsorted(edges, v)
        return idx < len(edges) and edges[idx] == v # type: ignore # idx is guaranteed to be an integer, but the type system cannot see


# method adapted from solution posted in https://discuss.datasciencedojo.com/t/how-can-an-array-of-integers-be-converted-into-a-binary-matrix/1056/2
def numbers_to_bit_matrix(arr: npt.NDArray[np.int32], bit_count: int) -> npt.NDArray[np.uint8]:
    used_mask = __mask[0:bit_count]

    return (np.bitwise_and(arr[:, None], used_mask) > 0).reshape(-1, bit_count).astype(np.uint8)

#Naive binary circuit to check adjancency
def build_adj_circuit(graph: Graph, output: int | None = None, circuit: QuantumCircuit | None = None ) -> Tuple[QuantumCircuit, list[int]] :
    try:
        bit_count = int(np.log2(graph.n).astype(dtype=np.uint32, casting='same_value'))
    except ValueError:
        raise Exception("graph n count must be a number which is a power of 2.")

    if circuit == None:
        u_reg, v_reg = QuantumRegister(bit_count, "u"), QuantumRegister(bit_count, "v")
        out = QuantumRegister(1, "output")
        circuit = QuantumCircuit(u_reg, v_reg, out, ClassicalRegister(1))
        combined_ctrl_qubits = list(u_reg) + list(v_reg)
        output = bit_count * 2
        ret_val = (circuit, list(range(0,6)))
    else:
        assert circuit.num_qubits >= bit_count * 2 + 1
        if output == None:
            output = circuit.num_qubits - 1
        qubits_range = np.arange(0, circuit.num_qubits)
        combined_ctrl_qubits: list[int] = qubits_range[qubits_range != output][:bit_count * 2].tolist()
        ret_val = (circuit, combined_ctrl_qubits)

    vertex_bits = numbers_to_bit_matrix(np.arange(0,graph.n, dtype=np.int32), bit_count)

    gates = []

    for idx in range(0, graph.n):
        u_bits: np.ndarray = vertex_bits[idx]
        #u_sign = np.nonzero(u_bits) # get the indices of every bit set to one

        edge_arr = graph.adj_list[idx]
        edge_arr = edge_arr[:edge_arr[-1]]
        edge_binary = numbers_to_bit_matrix(edge_arr, bit_count)

        u_ctrl_str = "".join(map(str, u_bits))
        
        for v_bits in edge_binary:
            v_ctrl_str = "".join(map(str, v_bits))

            ctrl_str = (u_ctrl_str + v_ctrl_str)[::-1]
            #Doing it this way is REALLY inefficient, the circuit depth will be crazy, and wont scale on a real quantum computer due to accumulated noise.
            gates.append({'instruction': MCXGate(num_ctrl_qubits=bit_count*2, ctrl_state=ctrl_str),'qargs':combined_ctrl_qubits + [output]})

    for gate in gates:
        circuit.append(**gate)
        circuit.barrier()    
    circuit.measure(output, 0)
    for gate in reversed(gates):
        circuit.append(**gate) 
        circuit.barrier()
    return ret_val


def adjancency_of_edge(graph: Graph, circuit: QuantumCircuit, output: int, node: int, bit_count: int, control: list[int], inputs: list[list[int]] ):
    u_bits = numbers_to_bit_matrix(np.asarray(node), bit_count)

    gates = []

        #u_sign = np.nonzero(u_bits) # get the indices of every bit set to one

    edge_arr = graph.adj_list[node]
    edge_arr = edge_arr[:edge_arr[-1]]
    edge_binary = numbers_to_bit_matrix(edge_arr, bit_count)

    u_ctrl_str = "".join(map(str, u_bits))
        
    for v_bits in edge_binary:
        v_ctrl_str = "".join(map(str, v_bits))

        ctrl_str = (u_ctrl_str + v_ctrl_str)[::-1]
        for input in inputs:
            combined_ctrl_qubits = control + input
            gates.append({'instruction': MCXGate(num_ctrl_qubits=bit_count*2, ctrl_state=ctrl_str),'qargs':combined_ctrl_qubits + [output]})

    for gate in gates:
        circuit.append(**gate)
        circuit.barrier()    

def invert_adjancency_of_edge(graph: Graph, circuit: QuantumCircuit, output: int, node: int, bit_count: int, control: list[int], inputs: list[list[int]] ):
    u_bits = numbers_to_bit_matrix(np.asarray(node), bit_count)

    gates = []

        #u_sign = np.nonzero(u_bits) # get the indices of every bit set to one

    edge_arr = graph.adj_list[node]
    edge_arr = edge_arr[:edge_arr[-1]]
    edge_binary = numbers_to_bit_matrix(edge_arr, bit_count)

    u_ctrl_str = "".join(map(str, u_bits))
        
    for v_bits in edge_binary:
        v_ctrl_str = "".join(map(str, v_bits))

        ctrl_str = (u_ctrl_str + v_ctrl_str)[::-1]
        for input in inputs:
            combined_ctrl_qubits = control + input
            gates.append({'instruction': MCXGate(num_ctrl_qubits=bit_count*2, ctrl_state=ctrl_str),'qargs':combined_ctrl_qubits + [output]})

    for gate in reversed(gates):
        circuit.append(**gate)
        circuit.barrier()    

# I'm assuming A and B are bit lists here
def init_and_run_adjacent_circuit(graph: Graph, circuit: QuantumCircuit, A: npt.NDArray[np.uint8] | list[int], B: npt.NDArray[np.uint8] | list[int], output: int):
    try:
        bit_count = int(np.log2(graph.n).astype(dtype=np.uint32, casting='same_value'))
    except ValueError:
        raise Exception("graph n count must be a number which is a power of 2.")

    circuit.clear()

    init_state = '0' + ("".join(map(str, A)) + "".join(map(str, B)))[::-1] # reversed again, "first" zero is actually to zero out the output register

    circuit.initialize(init_state) 
    
    _, input_qubits = build_adj_circuit(graph, output, circuit)

    simulator = AerSimulator()
    t_circuit = transpile(circuit, simulator)
    result = simulator.run(t_circuit, shots=10).result()

    return result.get_counts(t_circuit)

def rerun_adjacent_circuit(circuit: QuantumCircuit, A: npt.NDArray[np.uint8] | list[int], B: npt.NDArray[np.int8]):
    gates = [n for n in reversed(circuit.data) if n.name != 'initialize']
    init_state = '0' + ("".join(map(str, A)) + "".join(map(str, B)))[::-1]
    gates.append(Initialize(init_state)) # recreate initial state with new input

    circuit.clear()
    circuit.append(reversed(gates))

    simulator = AerSimulator()
    t_circuit = transpile(circuit, simulator)
    result = simulator.run(t_circuit, shots=10).result()

    return result.get_counts(t_circuit)

# OR implementation without using the builtin OR, or requiring auxiliaries.
def or_gate(circuit: QuantumCircuit, inputs: list[int], output: int):
    circuit.x(inputs) # Invert for AND
    circuit.mcx(inputs, output) # only an initial qubit state of |0> on all inputs will cause this to flip, which will cancel out the incoming NOT on the output gate, which effectively produces an OR-like output.
    circuit.x(inputs + [output])


# 
def multi_equiv(circuit: QuantumCircuit, ctrl_bits: npt.NDArray[np.uint8], control: npt.NDArray[np.uint32], inputs: npt.NDArray[np.uint32], auxiliary: int):
    '''
    test if at least one of the input bitstrings and control contains pattern
    Effectively an expanded OR. 
    
    :param circuit: target circuit
    :type circuit: QuantumCircuit
    :param pattern: ctrl_state pattern for the desired bitstring. Should already be reversed before passing it to this function
    :type pattern: str
    :param control: static qbits that the inputs should be compared against(called B in the case of part 1.3)
    :type control: list[int]
    :param inputs: list of qbits compared against control(called A in part 1.3)
    :type inputs: list[list[int]]
    :param auxiliary: Auxiliary qbit to control OR-state(can be output)
    :type auxiliary: int
    :param output: output qbit
    :type output: int
    '''
    pattern = ("".join(map(str, ctrl_bits)) * 2)[::-1]
    circuit.x(auxiliary)
    ctrl_state = '1' + pattern*2
    for input in inputs:
        # We are also 
        gate = MCXGate(len(pattern) + 1, ctrl_state="1" + pattern)
        circuit.append(gate, control + input + [auxiliary] )

    circuit.x(auxiliary)

def xor_gate(circuit: QuantumCircuit, inputs, output: int):
    circuit.mcx(inputs[:-1], inputs[-1]) # Invert for AND
    circuit.cx(inputs[-1], output) # only an initial qubit state of |0> on all inputs will cause this to flip, which will cancel out the incoming NOT on the output gate, which effectively produces an OR-like output.
    circuit.mcx(inputs[:-1], inputs[-1]) # Revert


#Create Dominating sets, expects len(B) + len(A) registers each holding logv2 n bits, plus 2 qubits reserved for output/auxiliary
def init_run_dominating_set(graph: Graph, circuit: QuantumCircuit, A: npt.NDArray[np.uint32] | list[int], B: npt.NDArray[np.uint8], auxiliary: int, output: int):
    try:
        bit_count = int(np.log2(graph.n).astype(dtype=np.uint32, casting='same_value'))
    except ValueError:
        raise Exception("graph n count must be a number which is a power of 2.")
    
    multi_equiv(circuit, B, np.arange(0, bit_count, dtype=np.uint32), np.arange(bit_count, bit_count * len(A), dtype=np.uint32), auxiliary)






def main():
    G = Graph(4)
    G.add_edge(0, 1)
    G.add_edge(0, 3)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    G.print()

    edges = numbers_to_bit_matrix(np.array([2,3], np.int32), 2)
    circuit = QuantumCircuit(5,1)
    #circuit, list = build_adj_circuit(G, -1)
    print(init_and_run_adjacent_circuit(G, circuit, edges[0], edges[1], 4))
    print(circuit)
    

if __name__=="__main__":
    main()


        
