#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 19:14:16 2022

@author: Shawn
"""

import numpy as np
import pandas as pd
import os
import networkx as nx
import community as community_louvain
import networkx.algorithms.tree as tree
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


file = '/Users/huchangguo/Desktop'
df = pd.read_csv(os.path.join(file,'person_knows_person(1).csv'))

#set for 50
droprows = []
for i in range(df.shape[0]):
    n1,n2 = df['Person.id|Person.id'][i].split('|')
    n1 = int(n1)
    n2 = int(n2)
    if n1 > 50 or n2 > 50:
        droprows.append(i)
df = df.drop(droprows)
df = df.reset_index(drop=True)


nlist = []
elist = []
g = nx.Graph()
for i in df['Person.id|Person.id']:
    n1,n2 = i.split('|')
    n1 = int(n1)
    n2 = int(n2)
    if n1 < 50 and n2 < 50:
        nlist.append(n1)
        nlist.append(n2)
        elist.append((n1,n2))
nlist = set(nlist)
g.add_nodes_from(nlist)
g.add_edges_from(elist)
nx.draw(g, with_labels = True, edge_color='grey', node_color='yellow')


startnode = np.sort(g.nodes())[0]
endnode = np.sort(g.nodes())[-1]
startnode = random.choice(np.sort(g.nodes))
endnode = random.choice(np.sort(g.nodes)) 


d_path = nx.dijkstra_path(g,startnode,endnode)
print(d_path)

nodecolor = []
for node in g.nodes():
    if node in d_path:
        nodecolor.append('red')
    else:
        nodecolor.append('green')
edgecolor = []
for edge in g.edges():
    n1, n2 = edge
    if n1 in d_path and n2 in d_path:
        edgecolor.append('red')
    else:
        edgecolor.append('grey')
nx.draw(g,with_labels = True,edge_color = edgecolor, node_color = nodecolor)


#use A* algorithm to generate the shortest path
a_path = nx.astar_path(g, startnode, endnode, heuristic=None, weight="weight")
a_length = nx.astar_path_length(g, startnode, endnode, heuristic=None, weight="weight")
print(a_path)
print(a_length)

nodecolor  = []
for node in g.nodes():
    if node in a_path:
        nodecolor.append('red')
    else:
        nodecolor.append('green')
edgecolor = []
for edge in g.edges():
    n1, n2 = edge
    if n1 in a_path and n2 in a_path:
        edgecolor.append('red')
    else:
        edgecolor.append('grey')
nx.draw(g,with_labels = True,edge_color = edgecolor, node_color = nodecolor)

primmst = tree.minimum_spanning_edges(g, algorithm='prim', data=False)
primedgelist = list(primmst)
print(primedgelist)

prim_g = nx.Graph()
prim_g.add_nodes_from(g.nodes)
prim_g.add_edges_from(primedgelist)
nx.draw(prim_g,with_labels = True, edge_color = 'grey', node_color = 'blue')

kruskalmst = tree.minimum_spanning_edges(g, algorithm='kruskal', data=False)
kruskaledgelist = list(kruskalmst)
print(kruskaledgelist)

kruskal_g = nx.Graph()
kruskal_g.add_nodes_from(g.nodes)
kruskal_g.add_edges_from(kruskaledgelist)
nx.draw(kruskal_g,with_labels = True, edge_color = 'grey', node_color = 'yellow')


pr = nx.pagerank(g, alpha=0.4)
print(pr)
#Use the size of the cricle to shows the importance of the node
nodesizepr = []
for node in g.nodes():
    nodesize = 10000*pr[node]
    nodesizepr.append(nodesize)
nx.draw(g,with_labels = True, edge_color = 'grey', node_color = 'pink',node_size = nodesizepr)


#HITS algorithm
hits = nx.hits(g, max_iter = 50, normalized = True)
h,a = hits

print(a)
nodesizea = []
for node in g.nodes():
    nodesize = 10000*a[node]
    nodesizea.append(nodesize)
nx.draw(g,with_labels = True, edge_color = 'grey', node_color = 'red',node_size = nodesizea)

print(h)
nodesizeh = []
for node in g.nodes():
    nodesize = 10000*h[node]
    nodesizeh.append(nodesize)
nx.draw(g,with_labels = True, edge_color = 'grey', node_color = 'green',node_size = nodesizeh)


nodesizehits = []
for node in g.nodes():
    nodesizehits.append(10000*(a[node]+h[node])/2)
nx.draw(g,with_labels = True, edge_color = 'grey', node_color = 'yellow',node_size = nodesizehits)





# first one：Louvain for community detection
partition = community_louvain.best_partition(g)
catagory = set([v for k,v in partition.items()])
for i in catagory:
    print('For',i,'cluster(s), the nodes are',[k for k,v in partition.items() if v == i])
#visualize
pos = nx.spring_layout(g)
cmap = cm.get_cmap('summer', max(partition.values()) + 1)
nx.draw_networkx_nodes(g, pos, partition.keys(), node_size=50,cmap=cmap, label = True, node_color=list(partition.values()))
nx.draw_networkx_edges(g, pos, alpha=0.5)
nx.draw_networkx_labels(g, pos,font_size=8)
plt.show()



