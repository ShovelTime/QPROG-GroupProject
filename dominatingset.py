from typing import Any, Tuple, Union, Optional, List, Sequence
import numpy as np
import numpy.ma as ma
from numpy.typing import NDArray, DTypeLike
import math
import time
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.circuit.library import Initialize
from qiskit.circuit import Instruction
from qiskit.circuit.library import MCXGate

from itertools import combinations

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
    
    def get_edge_of(self, u:int) -> NDArray[np.uint32]:
        edge_set = self.adj_list[u, :]
        set_len = edge_set[-1]
        return edge_set[:set_len]

    
    #Get the minimum bit count to represent all n values in binary
    def get_bit_count(self) -> int:
        return int(np.array(np.ceil(np.log2(self.n))).astype(np.uint32)) # type: ignore
    
    def get_bit_count_checked(self) -> int:
        try:
            return int(np.array(np.log2(self.n)).astype(np.uint32, casting='same_value'))# type: ignore
        except ValueError:
            raise Exception("graph n count must be a number which is a power of 2.")



############################################################
###################### HELPER METHODS ######################
############################################################

# method adapted from solution posted in https://discuss.datasciencedojo.com/t/how-can-an-array-of-integers-be-converted-into-a-binary-matrix/1056/2
def numbers_to_bit_matrix(arr: NDArray[np.uint32], bit_count: int) -> NDArray[np.uint8]:
    """
    Converts a list of numbers into a bit matrix
    (rows are the binary representation in Little-endian).
    """
    arr = np.atleast_1d(arr)
    used_mask = __mask[0:bit_count]

    return (np.bitwise_and(arr[:, None], used_mask) > 0).reshape(-1, bit_count).astype(np.uint8)

def bit_matrix_to_numbers(bit_matrix: NDArray[np.uint8]) -> NDArray[np.int32]:
    """
    Converts a matrix of bits back to an array of integers.
    (rows are numbers, cols are bits LSB-first) 
    """
    bit_count = bit_matrix.shape[0]
    weights = np.array([1 << i for i in range(bit_count)], dtype=np.int32)
    return bit_matrix.dot(weights)

#Basically X gates over only bits set to one
def initialize(circuit: QuantumCircuit, bit_pattern: NDArray[np.uint8], registers: NDArray[np.uint32]):
    indices = np.nonzero(bit_pattern.flatten())[0] # find non zero bits to apply the X gate to to
    if len(indices) > 0 : 
        circuit.x(registers[indices], label='initialize')

#Alias for initialize, since the inverse of it is the same.
def deinitialize(circuit: QuantumCircuit, bit_pattern: NDArray[np.uint8], registers: NDArray[np.uint32]):
    initialize(circuit, bit_pattern, registers)

# OR implementation without using the builtin OR, or requiring auxiliaries.
def or_gate(circuit: QuantumCircuit, inputs: list[int], output: int):
    circuit.x(inputs) # Invert for AND
    circuit.mcx(inputs, output) # only an initial qubit state of |0> on all inputs will cause this to flip, which will cancel out the incoming NOT on the output gate, which effectively produces an OR-like output.
    circuit.x(inputs + [output])

def xor_gate(circuit: QuantumCircuit, inputs: list[int], output: int):
    circuit.mcx(inputs[:-1], inputs[-1]) # Invert for AND
    circuit.cx(inputs[-1], output) # if every bit was 1, then inputs[-1] will be zero, causing no flip
    circuit.mcx(inputs[:-1], inputs[-1]) # Revert

#Based on lecture 5 slides
# TODO: Understand what is happening here
def diffusion(circuit: QuantumCircuit, registers: NDArray[np.uint32]):
    circuit.h(registers)
    circuit.x(registers)
    circuit.barrier()

    #W Operation
    circuit.h(registers[-1])
    circuit.mcx(registers[:-1].tolist(), registers[-1])
    circuit.h(registers[-1])

    circuit.barrier()
    circuit.x(registers)
    circuit.h(registers)

def run_circuit(circuit: QuantumCircuit):
    simulator = AerSimulator()
    t_circuit = transpile(circuit, simulator)
    result = simulator.run(t_circuit, shots=10).result()

    return result.get_counts(t_circuit)


############################################################
##################### QUANTUM CIRCUITS #####################
############################################################

#Naive binary circuit to check adjancency
# Since we are not performing any latching this version does not need auxiliary qbits.
def build_adj_circuit(graph: Graph, output: Optional[int] = None, circuit: Optional[QuantumCircuit] = None):

    bit_count = graph.get_bit_count_checked()

    if circuit == None:
        u_reg, v_reg = QuantumRegister(bit_count, "u"), QuantumRegister(bit_count, "v")
        out = QuantumRegister(1, "output")
        circuit = QuantumCircuit(u_reg, v_reg, out, ClassicalRegister(1))
        combined_ctrl_qubits = list(u_reg) + list(v_reg)
        output = bit_count * 2
        #ret_val = (circuit, list(range(0,6)))
    else:
        assert circuit.num_qubits >= bit_count * 2 + 1
        if output == None:
            output = circuit.num_qubits - 1
        qubits_range = np.arange(0, circuit.num_qubits)
        combined_ctrl_qubits: list[int] = qubits_range[qubits_range != output][:bit_count * 2].tolist()
        #ret_val = (circuit, combined_ctrl_qubits)

    vertex_bits = numbers_to_bit_matrix(np.arange(0,graph.n, dtype=np.uint32), bit_count)

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
            gates.append({'instruction': MCXGate(bit_count*2, ctrl_state=ctrl_str),'qargs':combined_ctrl_qubits + [output]})

    for gate in gates:
        circuit.append(**gate)
        circuit.barrier()    
    circuit.measure(output, 0)
    #for gate in reversed(gates):
    #    circuit.append(**gate) 
    #    circuit.barrier()
    #return ret_val

# I'm assuming A and B are bit lists here
def Adj(graph: Graph, circuit: QuantumCircuit, A: NDArray[np.uint8], B: NDArray[np.uint8], output: int):

    bit_count = graph.get_bit_count_checked()

    circuit.clear()

    initialize(circuit, np.concat([B, A]), np.arange(0, bit_count * 2, dtype=np.uint32)) 
    
    build_adj_circuit(graph, output, circuit)

    return run_circuit(circuit)



# Version specifically for Grover, which is why there is an auxiliary
# Auxiliary prevents further computations if it is set to one
#def multi_equiv(circuit: QuantumCircuit, ctrl_bits: NDArray[np.uint8], control: NDArray[np.uint32], inputs: NDArray[np.uint32], output: int, auxiliary: int | None = None):
#    '''
#    test if at least one of the input bitstrings and control contains pattern.
#    
#    :param circuit: target circuit
#    :type circuit: QuantumCircuit
#    :param pattern: ctrl_state pattern for the desired bitstring. Should already be reversed before passing it to this function
#    :type pattern: str
#    :param control: static qbits that the inputs should be compared against(called B in the case of part 1.3)
#    :type control: list[int]
#    :param inputs: list of qbits compared against control(called A in part 1.3)
#    :type inputs: list[list[int]]
#    :param auxiliary: Auxiliary qbit to control OR-state(can be output)
#    :type auxiliary: int
#    :param output: output qbit
#    :type output: int
#    '''
#    bit_count = len(ctrl_bits)
#    pattern = ("".join(map(str, ctrl_bits)) * 2)[::-1]
#
#    if auxiliary is not None:
#        ctrl_state = "0" * len(auxiliary) + pattern
#        extra_qubits = np.concat([auxiliary, [np.uint32(output)]], dtype=np.uint32)
#    else:
#        ctrl_state = pattern
#        extra_qubits = [output]
#    #circuit.x(auxiliary)
#    for input in inputs.reshape((-1, bit_count)):
#        gate = MCXGate(len(ctrl_state), ctrl_state=ctrl_state)
#        qargs = np.concat([control, input, extra_qubits], dtype=np.uint32)
#        circuit.append(gate, qargs.tolist())
#
#    #circuit.x(auxiliary)
#    circuit.barrier()

#def invert_multi_equiv(circuit: QuantumCircuit, ctrl_bits: NDArray[np.uint8], control: NDArray[np.uint32], inputs: NDArray[np.uint32], output: int, auxiliary: int | None = None):
#    bit_count = len(ctrl_bits)
#    pattern = ("".join(map(str, ctrl_bits)) * 2)[::-1]
#
#    if auxiliary is not None:
#        ctrl_state = "0" + pattern
#        extra_qubits = np.concat([auxiliary, [output]])
#    else:
#        ctrl_state = pattern
#        extra_qubits = [output]
#    #circuit.x(auxiliary)
#    ctrl_state = "0" + pattern
#    for input in reversed(inputs.reshape((-1, bit_count))):
#        gate = MCXGate(len(ctrl_state), ctrl_state=ctrl_state)
#        qargs = np.concat([control, input, extra_qubits], dtype=np.uint32)
#        circuit.append(gate, qargs.tolist())
#
#    #circuit.x(auxiliary)
#    circuit.barrier()

# Version specifically for Grover, which is why there is an auxiliary
# Auxiliary prevents further computations if it is set to one
def __adjancency_of_edge(graph: Graph, circuit: QuantumCircuit, 
                       auxiliary:NDArray[np.uint32], output: int, 
                       node: int, bit_count: int, 
                       control: NDArray[np.uint32], inputs: NDArray[np.uint32] ):
    u_bits = numbers_to_bit_matrix(np.uint32(node), bit_count).flatten()

    assert len(auxiliary) >= 2
    used_auxiliaries = auxiliary[:2]
    extra_registers = np.concat([used_auxiliaries, np.array([output], dtype=np.uint32)])
    extra_pattern = "10"

    gates = []

    edge_arr = graph.adj_list[node]
    edge_arr = edge_arr[:edge_arr[-1]]
    edge_binary = numbers_to_bit_matrix(edge_arr, bit_count)

    u_ctrl_str = "".join(map(str, u_bits))
    for input in inputs.reshape((-1, bit_count)):
        input_gates = []
        combined_ctrl_qubits = np.concatenate([control, input])
        combined_ctrl_qubits = np.concatenate([combined_ctrl_qubits, extra_registers])
        input_gates.append({'instruction': MCXGate(num_ctrl_qubits=bit_count * 2 + len(used_auxiliaries),ctrl_state= extra_pattern + (u_ctrl_str + u_ctrl_str)[::-1]), 'qargs':combined_ctrl_qubits.tolist()})
        for v_bits in edge_binary:
            v_ctrl_str = "".join(map(str, v_bits))
            ctrl_str = (u_ctrl_str + v_ctrl_str)[::-1]
            input_gates.append({'instruction': MCXGate(num_ctrl_qubits=bit_count*2 + len(used_auxiliaries), ctrl_state=extra_pattern + ctrl_str),'qargs':combined_ctrl_qubits.tolist()})
        gates.append(input_gates)

    circuit.x(auxiliary[1])
    for input_gates in gates:
        for gate in input_gates:
            circuit.append(**gate)
        
        circuit.cx(auxiliary[0], auxiliary[1]) 
        circuit.x(output)

        circuit.ccx(output, auxiliary[1], auxiliary[0]) # Set 'Fail' Flag if result of dominating_node is False
        circuit.ccx(output, auxiliary[0], auxiliary[1]) # set output to zero if Fail is set
        circuit.ccx(output, auxiliary[1], auxiliary[0]) # Set 'Fail' Flag if result of dominating_node is False

        circuit.x(output)
        circuit.cx(auxiliary[0], auxiliary[1]) 
        circuit.barrier()   

def __invert_adjancency_of_edge(graph: Graph, circuit: QuantumCircuit, 
                              auxiliary: int, output: int, 
                              node: int, bit_count: int, 
                              control: NDArray[np.uint32], inputs: NDArray[np.uint32] ):
    u_bits = numbers_to_bit_matrix(np.uint32(node), bit_count).flatten()

    assert len(auxiliary) >= 2
    used_auxiliaries = auxiliary[:2]
    extra_registers = np.concat([used_auxiliaries, np.array([output], dtype=np.uint32)])
    extra_pattern = "00"

    gates = []

    edge_arr = graph.adj_list[node]
    edge_arr = edge_arr[:edge_arr[-1]]
    edge_binary = numbers_to_bit_matrix(edge_arr, bit_count)

    u_ctrl_str = "".join(map(str, u_bits))
        
    for input in inputs.reshape((-1, bit_count)):
        input_gates = []
        combined_ctrl_qubits = np.concatenate([control, input])
        combined_ctrl_qubits = np.concatenate([combined_ctrl_qubits, extra_registers])
        for v_bits in edge_binary:
            v_ctrl_str = "".join(map(str, v_bits))
            ctrl_str = (u_ctrl_str + v_ctrl_str)[::-1]
            input_gates.append({'instruction': MCXGate(num_ctrl_qubits=bit_count*2 + len(used_auxiliaries), ctrl_state=extra_pattern + ctrl_str),'qargs':combined_ctrl_qubits.tolist()})
        input_gates.append({'instruction': MCXGate(num_ctrl_qubits=bit_count * 2 + len(used_auxiliaries),ctrl_state= extra_pattern + (u_ctrl_str + u_ctrl_str)[::-1]), 'qargs':combined_ctrl_qubits.tolist()})
        gates.append(input_gates)

    for input_gates in reversed(gates):
        circuit.cx(auxiliary[0], auxiliary[1]) 
        circuit.x(output)

        circuit.ccx(output, auxiliary[1], auxiliary[0]) # Set 'Fail' Flag if result of dominating_node is False
        circuit.ccx(output, auxiliary[0], auxiliary[1]) # set output to zero if Fail is set
        circuit.ccx(output, auxiliary[1], auxiliary[0]) # Set 'Fail' Flag if result of dominating_node is False

        circuit.x(output)
        circuit.cx(auxiliary[0], auxiliary[1]) 
        circuit.barrier() 
        for gate in reversed(input_gates):
            circuit.append(**gate)
    circuit.x(auxiliary[1])
        
   

#Create Dominating sets, expects len(B) + len(A) registers each holding logv2 n bits, plus at least 1 qubits reserved for output/auxiliary
def dominating_node(graph: Graph, circuit: QuantumCircuit, 
                           A: NDArray[np.uint8], 
                           B: NDArray[np.uint8], 
                           auxiliary: NDArray[np.uint32],
                           output: int, 
                           B_register: NDArray[np.uint32] | None = None,
                           A_registers: NDArray[np.uint32] | None = None):
    bit_count = graph.get_bit_count_checked()
    
    node = np.asarray(bit_matrix_to_numbers(B))[0]
    if B_register is None: B_register = np.arange(0, bit_count, dtype=np.uint32)
    if A_registers is None: A_registers = np.arange(bit_count, bit_count * (len(A) + 1), dtype=np.uint32)

    assert output not in auxiliary and not np.isin(auxiliary, np.concat([B_register, A_registers])).any()
    #multi_equiv(circuit, B, B_register, A_registers, output=auxiliary[0], auxiliary=auxiliary[1])
    __adjancency_of_edge(graph, circuit, auxiliary, output, node, bit_count, B_register, A_registers)

def invert_dominating_node(graph: Graph, circuit: QuantumCircuit, 
                           A: NDArray[np.uint8], 
                           B: NDArray[np.uint8], 
                           auxiliary: NDArray[np.uint32],
                           output: int, 
                           B_register: NDArray[np.uint32] | None = None,
                           A_registers: NDArray[np.uint32] | None = None):
    bit_count = graph.get_bit_count_checked()
    
    node = np.asarray(bit_matrix_to_numbers(B))[0]
    if B_register is None: B_register = np.arange(0, bit_count, dtype=np.uint32)
    if A_registers is None: A_registers = np.arange(bit_count, bit_count * (len(A) + 1), dtype=np.uint32)

    circuit.cx(output, auxiliary[0])
    __invert_adjancency_of_edge(graph, circuit, auxiliary, output, node, bit_count, B_register, A_registers)
    circuit.cx(output, auxiliary[0])
    #invert_multi_equiv(circuit, B, B_register, A_registers, output=output, auxiliary=auxiliary)

def Dominated(graph: Graph, circuit: QuantumCircuit, A: NDArray[np.uint8], B: NDArray[np.uint8], auxiliary: NDArray[np.uint32], output:int):
    bit_count = graph.get_bit_count_checked()

    assert output not in auxiliary and len(auxiliary) >= 2
    assert circuit.num_qubits >= A.shape[0] + len(B) + len(auxiliary) + 1

    circuit.clear()

    B_register = np.arange(0, bit_count, dtype=np.uint32)
    A_registers = np.arange(bit_count, bit_count * (len(A) + 1), dtype=np.uint32)
    initialize(circuit, B, B_register)
    initialize(circuit, A, A_registers)
    circuit.x(output)
    circuit.barrier()
    __dominating_node(graph, circuit, B=B, auxiliary=auxiliary, output=output, B_register=B_register, A_registers=A_registers)
    circuit.measure(output, 0)
    #__invert_dominating_node(graph, circuit, B=B, auxiliary=auxiliary, output=output, B_register=B_register, A_registers=A_registers)

    return run_circuit(circuit)



# TODO: Inversion without losing output?
def dominating_set(graph: Graph, circuit: QuantumCircuit, A: NDArray[np.uint8], auxiliary: NDArray[np.uint32], output: int):
    bit_count = graph.get_bit_count_checked()
    
    if output in auxiliary:
        raise Exception("Output cannot be an auxiliary qubit for this implementation!")
    if len(auxiliary) < 3:
        raise Exception("dominating_set() needs at least 3 auxiliary qubits!")
    
    
    vertex_bits = numbers_to_bit_matrix(np.array(np.arange(0, graph.n), np.uint32), bit_count)

    assert circuit.num_qubits >= bit_count * len(A) + 3

    register_range = np.arange(0, bit_count*len(A) + 1, dtype=np.uint32)

    assert output not in register_range and not np.isin(auxiliary, register_range).any()

    initialize(circuit, vertex_bits.flatten(), register_range[bit_count:]) # Initialize A's qubits
    circuit.x(auxiliary[2])
    for u_bits in vertex_bits:
        initialize(circuit, u_bits, register_range[:bit_count])

        dominating_node(graph, circuit, A, u_bits, auxiliary=auxiliary[1:], output=auxiliary[0])

        #circuit.cx(auxiliary[1], auxiliary[2]) 
        #circuit.x(auxiliary[0])
#
        #circuit.ccx(auxiliary[0], auxiliary[2], auxiliary[1]) # Set 'Fail' Flag if result of dominating_node is False
        #circuit.ccx(auxiliary[0], auxiliary[1], auxiliary[2]) # set output to zero if Fail is set
        #circuit.ccx(auxiliary[0], auxiliary[2], auxiliary[1]) # Revert if needed
#
        #circuit.x(auxiliary[0]) 
        #circuit.cx(auxiliary[1], auxiliary[2]) 

        invert_dominating_node(graph, circuit, A, u_bits, auxiliary[1:], auxiliary[0])

        deinitialize(circuit, u_bits, register_range[:bit_count])

    circuit.cx(auxiliary[2], output)


def __dominating_node(graph: Graph, circuit: QuantumCircuit, 
                           B : NDArray[np.uint8],
                           auxiliary: NDArray[np.uint32],
                           output : int,
                           B_register: NDArray[np.uint32],
                           A_registers: NDArray[np.uint32]):
    bit_count = graph.get_bit_count_checked()
    
    node = bit_matrix_to_numbers(B)

    __adjancency_of_edge(graph, circuit, auxiliary=auxiliary, output=output, node=node, bit_count=bit_count, control=B_register, inputs=A_registers)


def __invert_dominating_node(graph: Graph, circuit: QuantumCircuit, 
                           B : NDArray[np.uint8],
                           auxiliary: NDArray[np.uint32],
                           output : int,
                           B_register: NDArray[np.uint32],
                           A_registers: NDArray[np.uint32]):
    try:
        bit_count = np.array(np.log2(graph.n)).astype(np.uint32, casting='same_value')[0]
    except ValueError:
        raise Exception("graph n-count must be a number which is a power of 2.")
    
    node = np.asarray(bit_matrix_to_numbers(B))[0]
    circuit.cx(auxiliary[0], auxiliary[1])
    __invert_adjancency_of_edge(graph, circuit, auxiliary=auxiliary, output=output, node=node, bit_count=bit_count, control=B_register, inputs=A_registers)
    circuit.cx(auxiliary[0], auxiliary[1])
    invert_multi_equiv(circuit, B, B_register, A_registers, auxiliary=auxiliary, output=output)

#dominating set variant which acts as a verifier circuit
def __dominating_verifier(graph: Graph, circuit: QuantumCircuit, 
                        auxiliary: NDArray[np.uint32], output: int,
                        A_registers: NDArray[np.uint32],
                        B_registers: NDArray[np.uint32]):
    try:
        bit_count = np.array(np.log2(graph.n)).astype(np.uint32, casting='same_value')[0]
    except ValueError:
        raise Exception("graph n-count must be a number which is a power of 2.")
    if len(auxiliary) < 3:
        raise Exception("dominating_set() needs at least 3 auxiliary qubits!")
    
    
    vertex_bits = numbers_to_bit_matrix(np.array(np.arange(0, graph.n), np.uint32), bit_count)

    assert circuit.num_qubits >= bit_count * A_registers + 3
    assert auxiliary not in np.concatenate([A_registers, B_registers])

    circuit.x(auxiliary[2])
    for u_bits in vertex_bits:
        initialize(circuit, u_bits, B_registers)

        __dominating_node(graph, circuit, u_bits, auxiliary, B_registers, A_registers)

        circuit.cx(auxiliary[1], auxiliary[2]) 
        #circuit.cx(auxiliary[1], auxiliary[0]) # If the fail flag is set, we do not invert a fail result
        circuit.x(auxiliary[0])

        circuit.ccx(auxiliary[0], auxiliary[2], auxiliary[1]) # Set 'Fail' Flag if result of dominating_node is False
        circuit.ccx(auxiliary[0], auxiliary[1], auxiliary[2]) # set output to zero if Fail is set
        circuit.ccx(auxiliary[0], auxiliary[2], auxiliary[1]) # Revert if needed

        circuit.x(auxiliary[0]) 
        circuit.cx(auxiliary[1], auxiliary[2]) 

        __invert_dominating_node(graph, circuit, u_bits, auxiliary, B_registers, A_registers)

        deinitialize(circuit, u_bits, B_registers)
    
    circuit.cx(auxiliary[2], output)

# Inversion
    for u_bits in vertex_bits[::-1]:
        deinitialize(circuit, u_bits, B_registers)

        __invert_dominating_node(graph, circuit, u_bits, auxiliary, B_registers, A_registers)

        circuit.cx(auxiliary[1], auxiliary[2]) 
        #circuit.cx(auxiliary[1], auxiliary[0]) # If the fail flag is set, we do not invert a fail result
        circuit.x(auxiliary[0])

        circuit.ccx(auxiliary[0], auxiliary[2], auxiliary[1]) # Set 'Fail' Flag if result of dominating_node is False
        circuit.ccx(auxiliary[0], auxiliary[1], auxiliary[2]) # set output to zero if Fail is set
        circuit.ccx(auxiliary[0], auxiliary[2], auxiliary[1]) # Revert if needed

        circuit.x(auxiliary[0]) 
        circuit.cx(auxiliary[1], auxiliary[2]) 

        __dominating_node(graph, circuit, u_bits, auxiliary, B_registers, A_registers)

        initialize(circuit, u_bits, B_registers)
    circuit.x(auxiliary[2])
    
    

def grover(graph: Graph, circuit: QuantumCircuit, auxiliary: NDArray[np.uint32], output: int, k: int):
    try:
        bit_count = np.array(np.log2(graph.n)).astype(np.uint32, casting='same_value')[0]
    except ValueError:
        raise Exception("graph n-count must be a number which is a power of 2.")
    
    # + 1 to account for the B register containing the vertex input
    subset_size = k + 1 * bit_count
    iteration_count = int(np.pi / 4.0 * np.sqrt(graph.n ** k))
    
    if output in auxiliary:
        raise Exception("Output cannot be an auxiliary qubit for this implementation!")
    if len(auxiliary) < 2:
        raise Exception("grover_one() needs at least 2 auxiliary qubits!")
    
    assert circuit.num_qubits == subset_size + 3

    B = np.arange(0, bit_count, dtype=np.uint32)
    A = np.arange(bit_count, subset_size, dtype=np.uint32)

    circuit.h(A)

    for _ in range(iteration_count):
        circuit.barrier()
        __dominating_verifier(graph, circuit, auxiliary, output, A, B)
        diffusion(circuit, A)

    circuit.measure(A, np.arange(0, len(A)))
        
def classical_dominating_node(u: np.uint32, graph: Graph, A: NDArray[np.uint32]) -> np.bool | bool:
    adj_check = np.isin(A, graph.get_edge_of(u)) | (A == u)
    return adj_check.all()
#classical check of dominating set
def classical_dominating_set(graph:Graph, candidate_set: NDArray[np.uint32]) -> np.bool | bool:
    node_check = np.vectorize(classical_dominating_node, excluded=['graph', 'A'])
    results = node_check(np.arange(0, graph.n), graph, candidate_set)
    return results.all()



def main():
    pass



if __name__=="__main__":
    main()


        
