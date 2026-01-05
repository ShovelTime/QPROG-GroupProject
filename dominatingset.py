import numpy as np
import numpy.ma as ma

from typing import IO

class Graph:
    def __init__(self, n: int):
        '''
        Creates Graph stub, preallocates n sublists each holding n-1 elements, which allows for every vertices to be neighbors with every other.
        Empty neighbour slots are represented as the sentinel value -1, as vertices cannot be represented with a negative value.
        
        :param n: vertex count
        :type n: int
        '''
        self.n = n
        self.adj_list = np.full((n, n), -1, np.int32, order='F')
        self.adj_list[:, -1] = 1

    def set_number_vertices(self, n : int):
        '''
        Updates vertex count within the graph.
        
        :param self: Description
        :param n: Description
        :type n: int
        '''
        if n == self.n: return

        old_n = self.n
        old_lens = np.array(self.adj_list[:, -1])

        if n > self.n :
            diff = n - self.n
            self.adj_list = np.pad(self.adj_list, ((0, diff), (0, diff - 1)), "constant", constant_values=-1)
            self.adj_list[:old_n, -1] = old_lens
            self.adj_list[old_n:, -1] = 1
        else:
            self.adj_list = np.resize(self.adj_list, (n, n))
            self.adj_list[:, -1] = old_lens[:n]
    
    def add_edge(self, u: int, v: int):
        u_list = self.adj_list[u, :-1]
        u_len = self.adj_list[u,-1]
         
        if v not in u_list:
            if u_len >= u_list.shape[1]: raise Exception("Attempting to add a neighbour when every vertices should already be neighouring this vertices!")
            self.adj_list[u_len] = v
            self.adj_list[u,-1] = u_len + 1
        
        v_list = self.adj_list[v, :-1]
        v_len = self.adj_list[v,-1]
    
        if u not in v_list:
            if v_len >= u_list.shape[1]: raise Exception("Attempting to add a neighbour when every vertices should already be neighouring this vertices!")
            self.adj_list[v_len] = u
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

        


def main():
    test_graph = Graph(5)
    test_graph.print()


if __name__=="__main__":
    main()


        
