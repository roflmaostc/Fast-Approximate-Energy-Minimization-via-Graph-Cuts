#!/usr/bin/env python


'''
Minimal example to check performance of graph cuts
'''

import numpy as np
import maxflow as mf
import random
import time
import sys

def minimum_cut(graph, map):
    '''This functions calculates the minimum cut using the PyMaxflow library
       Input: graph, map and revmap
       output: list sorted in order of map with new label'''

    #first input is number of nodes, second is number of non-terminal edges
    graph_mf = mf.Graph[float](len(map),len(map))
    #add nodes
    nodes = graph_mf.add_nodes(len(map))

    #loop over adjacency matrix graph and add edges with weight
    start = time.time()
    for i in range(1, len(graph)-1):
        for j in range(i+1,len(graph)-1):
            if graph[i][j] > 0:
                graph_mf.add_edge(i-1,j-1, graph[i][j], graph[i][j])
    print("loop", time.time()-start)

    #add all the terminal edges
    for i in range(0,len(nodes)):
        graph_mf.add_tedge(nodes[i], graph[0][i+1], graph[-1][i+1])
   
    #computation intensive part; calculation of minimum cut
    start = time.time()
    flow = graph_mf.maxflow()
    print("max flow", time.time()-start)
    return [graph_mf.get_segment(nodes[i]) for i in range(0, len(nodes))]



def generate_graph(size = 100, dropout=0.5, edge_dropout=0.5):
    '''Graph which has the the same structure as our 2d graph'''

    #image without data 
    xlen = int((1-dropout)*size)
    ylen = int((1-dropout)*size) 

    #mapping image to graph
    map = {}
    
    graph = np.zeros((xlen*ylen+2,xlen*ylen+2), dtype=np.float)
   
    for i in range(1, len(graph)-1):
        graph[0,i] = random.randint(0,200)
        graph[i,0] = graph[0,i]
        graph[i,-1] = random.randint(0,200)
        graph[-1, i] = graph[i,-1]

    graph_pos = 1
    for x in range(0,xlen):
        for y in range(0, ylen):
            map[(y,x)] = graph_pos
            graph_pos += 1

    #add edges with random dropout
    for x in range(0, xlen):
        for y in range(0, ylen):
            for a,b in zip([1,0,-1,0],[0,1,0,-1]):
                if (x+a<xlen and x+a>=0) and (y+b<ylen and y+b>=0)\
                                      and random.random()>2*edge_dropout:
                    graph[map[(y,x)], map[(y+b,x+a)]] = random.randint(0, 200)
                    graph[map[(y+b,x+a)],map[(y,x)]] =  graph[map[(y,x)], map[(y+b,x+a)]]
    return graph, map


def simple_test():
    start = time.time()
    g, m   = generate_graph(100, 0, 0.9)
    print("Generating Graph with ", len(g)," nodes took", time.time()-start, " seconds")
    start = time.time()
    minimum_cut(g,m)
    print("Cutting Graph took", time.time()-start, " seconds")




def testing():
    print("pixels total \t time/s")
    for i in range(10,200, 10):
        g, m   = generate_graph(i, 0, 0)
        start = time.time()
        minimum_cut(g, m)
        stop = time.time()
        print(i, "\t \t", stop-start)


testing()
