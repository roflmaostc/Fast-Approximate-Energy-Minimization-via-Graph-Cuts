#!/usr/bin/env python

import numpy as np

import bidict as bi

def image_to_array(img):
    '''input: path to image
       output: array of grayscale
       reference: https://stackoverflow.com/questions/40727793/how-to-convert-a-grayscale-image-into-a-list-of-pixel-values
       '''
    from PIL import Image
    img = Image.open(img).convert('L')
    w, h = img.size

    data = list(img.getdata())
    data = [data[off:off+w] for off in range(0, w*h, w)]

    return data

def give_test_1d_image():
    '''returns a 1x10 pixel image'''
    return np.array([[5, 4, 7, 5, 5, 4, 4, 3, 4, 3]])#,\
                     # [5, 5, 5, 5, 5, 2, 3, 2, 5, 5]])
    # return np.array([[5,5,3],[5,7,5]])

def graph_to_xdot(graph, map, revmap):
    '''prints out graph in xdot format
       graph: graph array
       map: mapping from adjacency-matrix-pos to (x,y)
       map: mapping from (x,y) to adjacency-matrix-pos'''
    out = "strict graph G{\nnode[color=white, style=filled];\n"
    
    out += "\"alpha\"[color=red]\n "
    out += "\"beta\"[color=red]\n "
    for (y,x) in revmap:
        out += "\"x="+ str(x) + ";y=" + str(y)+ "\"[color=red]\n "

    for (y,x) in revmap:
        for j in range(1,len(graph)-1):
            if graph[revmap[(y,x)],j] != 0:
                y2, x2 = map[j]
                out +=  "\"x="+ str(x) + ";y=" + str(y) +"\"--" + "\"x="+ str(x2) + ";y=" + str(y2) +"\"\n" 
    for i in range(1, len(graph)-1):
        y, x = map[i]
        out +=  "\"x="+ str(x) + ";y=" + str(y) +"\"--" + "\"alpha\"\n" 
        out +=  "\"x="+ str(x) + ";y=" + str(y) +"\"--" + "\"beta\"\n" 

    out += "}"

    return out


def D_p(label, graph, x, y):
    '''Returns the quadratic difference between label and real intensity of pixel'''
    return (label-graph[y][x])**2




def create_graph(image, alpha, beta):
    '''Method creates special graph using adjacency matrix as representation.
       image: is the array in greyscale
       alpha: alpha label
       beta: beta label
    '''
    map = {}
    revmap = {}
    #loop over all pixels
    map_parameter = 1
    for i in range(len(image)):
        for j in range(len(image[0])):
            #extract pixel which have the wanted label
            if image[i][j] == alpha or image[i][j] == beta:
                map[map_parameter] = (i,j)
                revmap[(i,j)] = map_parameter
                map_parameter += 1

    n = len(map)
    #graph consists of all beta or alpha pixels
    #additionally 1 alpha and 1 beta node
    #alpha is at position 0, beta at n+1
    graph = np.zeros((n+2, n+2), dtype=np.float) 
    for i in range(1, n+1):
        #from alpha to all pixels
        graph[i, 0] = 1
        graph[0, i] = 1
        #from beta to all pixels
        graph[-1, i] = 2
        graph[i, -1] = 2

    #neighbour have edges too
    for key in map:
        x,y = map[key]
        #search top, down, left, right neighbour
        for a,b in zip([1,0,-1,0],[0,1,0,-1]):
            if (x+a,y+b) in revmap:
                graph[key, revmap[(x+a,y+b)]] = 1 
                graph[revmap[(x+a,y+b)], key] = 1 
        
    return graph, map, revmap


arr = image_to_array("../test.png")
graph, map, revmap = create_graph(arr, 0,3)
print(graph_to_xdot(graph, map, revmap))
