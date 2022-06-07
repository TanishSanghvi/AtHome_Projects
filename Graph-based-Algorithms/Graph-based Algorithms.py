#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:56:07 2021

@author: apple
"""

############################

# Trie Data Structure  - Insert, Search and Delete

class TrieNode:
    
    def __init__(self):
        self.children = [None] * 26
        self.status = False
        
class Trie:
    
    def __init__(self):
        self.root = self.get_node()
    
    def get_node(self):
        return TrieNode()
    
    def charIndex(self, ch):
        return ord(ch) - ord('a')
    
    def insert(self, key):
        
        root = self.root
        n = len(key)
        for i in range(n):
            index = self.charIndex(key[i])
            
            if not root.children[index]:
                root.children[index].get_node()
            root = root.children[index]
        
        root.status = True
        
    def search(self, key):
        
        root = self.root
        n = len(key)
        
        for i in range(n):
            index = self.charIndex(key[i])
            
            if not root.children[index]:
                return False
            root = root.children[index]
            
        return root.status
    
    def delete(self, key):
        
        root = self.root
        if search(key):
            for i in range(len(key)):
                index = self.charIndex(key[i])
                root = root.children[index]
            
            root.status = False
        
#####################################################

# BFS and DFS Traversal methods

from collections import defaultdict

class Graph:

    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)

    def bfs_traversal(self, val):
        
        visited = []
        queue = []
        
        visited.append(val)
        queue.append(val)
        
        while queue:
            s = queue.pop(0)
            print(s)
            
            for i in self.graph[s]:
                if i not in visited:
                    visited.append(i)
                    queue.append(i)
                    
    def dfs_traversal(self, val):
        
        visited = []
        queue = []
        
        visited.append(val)
        queue.append(val)
        
        while queue:
            s = queue.pop()
            print(s)
            
            for i in self.graph[s][::-1]:
                if i not in visited:
                    visited.append(i)
                    queue.append(i)
                    
    
    def dfs_util(self, val, visited):
        
        visited.append(val)
        
        for i in self.graph[val]:
            if i not in visited:
                self.dfs_util(i, visited)
        
    def dfs_traversal(self, val):
        
        visited = []
        
        self.dfs_util(val, visited)
        
                
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

g.bfs_traversal(2)
g.dfs_traversal(2)
        

#####################################################

# Dijkstra's Algotrithm
from heapq import heapify, heappush, heappop

def dijkistra(graph, src, dest):
    
    inf = sys.maxsize
    node_data = {'A':{'cost':inf, 'pred' : []},
                 'B':{'cost':inf, 'pred' : []},
                 'C':{'cost':inf, 'pred' : []},
                 'D':{'cost':inf, 'pred' : []},
                 'E':{'cost':inf, 'pred' : []},
                 'F':{'cost':inf, 'pred' : []}}
    
    visited = []
    temp = src
    node_data[temp]['cost']=0
    
    for i in range(len(node_data)-1):
        if temp not in visited:
            visited.append(temp)
            min_heap = []
            for j in graph[temp]:
                if j not in visited:
                    cost = node_data[temp]['cost'] + graph[temp][j]
                    if cost<node_data[j]['cost']:
                        node_data[j]['cost'] = cost
                        node_data[j]['pred'] = node_data[temp]['pred'] + list(temp)
                    heappush(min_heap, (node_data[j]['cost'], j))
        heapify(min_heap)
        temp = min_heap[0][1]
        
    print(node_data[dest]['cost'])
    print(node_data[dest]['pred'] + list(dest))
    
                        
    
graph = {
    'A':{'B':2, 'C':4},
    'B':{'A':2, 'C':1, 'D':8},
    'C':{'A':4, 'B':1, 'E':5, 'D':2},
    'D':{'B':8, 'C':2, 'E':11, 'F':22},
    'E':{'C':5, 'D':11, 'F':1},
    'F':{'D':22, 'E':1},
    }

src = 'A'
dest = 'F'

dijkistra(graph, source, destination)
        

#####################################################
    
# BFS shortest distance using Bellman
        
def Bellman(V, E, graph, src):
    
    dist = [sys.maxsize] * V
    
    dist[src] = 0
    
    for i in range(V-1):
        for j in range(E):
            if dist[graph[j][0]] + graph[j][2] < dist[graph[j][1]]:
                dist[graph[j][1]] = dist[graph[j][0]] + graph[j][2]
                
    
    for i in range(V):
        print(i, dist[i])
        

graph = [[0, 1, -1], [0, 2, 4], [1, 2, 3], [1, 3, 2], [1, 4, 2], [3, 2, 5], [3, 1, 1], [4, 3, -3]]          
        
V = 5
E = len(graph)

Bellman(V, E, graph, 0)


#####################################################

# Prim's Algorithm

def prims(vertices, G):

    mstMatrix = [[0 for column in range(vertices)] for row in range(vertices)]
    selected_vertices = [False for vertex in range(vertices)]
    
    positiveInf = sys.maxsize
    while False in selected_vertices:
        
        start = 0
        end = 0
        minimum = sys.maxsize
        
        for i in range(vertices):
            if selected_vertices[i]:
                for j in range(0+i, vertices):
                    if (not selected_vertices[j] and G[i][j]>0):
                        if G[i][j]<minimum:
                            minimum = G[i][j]
                            start, end = i,j
                            
        selected_vertices[end] = True
        mstMatrix[start][end] = minimum
        
        if minimum == positiveInf:
            mstMatrix[start][end] = 0
        
        mstMatrix[end][start] = mstMatrix[start][end]
        
    print(mstMatrix)      
    


G = [[0, 2, 0, 6, 0],
    [2, 0, 3, 8, 5],
    [0, 3, 0, 0, 7],
    [6, 8, 0, 0, 9],
    [0, 5, 7, 9, 0]]

prims(len(G), G)

#####################################################

#Kruskal's MST Algorithm

from collections import defaultdict

class Graph:
 
    def __init__(self, vertices):
        self.V = vertices  
        self.graph = []  
        
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
 
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])
 
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
 
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
 
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
 
    def KruskalMST(self):
 
        result = []  
        i = 0
        e = 0
 
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])
 
        parent = []
        rank = []
 
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
 
        while e < self.V - 1:
 
            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
         
        minimumCost = 0
        print ("Edges in the constructed MST")
        for u, v, weight in result:
            minimumCost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree" , minimumCost)
 
g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)

g.KruskalMST()


