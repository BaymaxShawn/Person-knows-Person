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

#用networkx创建graph，networkx 是一个包，创建图用
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
##可以 随机去取


#question1 用两个算法找最短路径
#找最短的路线 use dijkstra algorithm to generate the shortest path 
d_path = nx.dijkstra_path(g,startnode,endnode)
print(d_path)
#可视化最短的路线 visualize the shortest path
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


#第二个方法
#use A* algorithm to generate the shortest path
a_path = nx.astar_path(g, startnode, endnode, heuristic=None, weight="weight")
a_length = nx.astar_path_length(g, startnode, endnode, heuristic=None, weight="weight")
print(a_path)
print(a_length)
#可视化最短的路线 visualize the shortest path
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




#question2 
#第一个方式prim去找 MST， use Prim to find the MST
primmst = tree.minimum_spanning_edges(g, algorithm='prim', data=False)
primedgelist = list(primmst)
print(primedgelist)
#可视化新的图visualize the Prim MST
prim_g = nx.Graph()
prim_g.add_nodes_from(g.nodes)
prim_g.add_edges_from(primedgelist)
nx.draw(prim_g,with_labels = True, edge_color = 'grey', node_color = 'blue')

#第二个方式K去找MST， use Kruskal to find the MST
kruskalmst = tree.minimum_spanning_edges(g, algorithm='kruskal', data=False)
kruskaledgelist = list(kruskalmst)
print(kruskaledgelist)
#可视化新的图visualize the Kruskal MST
kruskal_g = nx.Graph()
kruskal_g.add_nodes_from(g.nodes)
kruskal_g.add_edges_from(kruskaledgelist)
nx.draw(kruskal_g,with_labels = True, edge_color = 'grey', node_color = 'yellow')





#question3
#Page rank
pr = nx.pagerank(g, alpha=0.4)
print(pr)
#用原点大小代表重要度，Use the size of the cricle to shows the importance of the node
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

#用两个score的平均值来画node size的大小
nodesizehits = []
for node in g.nodes():
    nodesizehits.append(10000*(a[node]+h[node])/2)
nx.draw(g,with_labels = True, edge_color = 'grey', node_color = 'yellow',node_size = nodesizehits)






#question4
#用第一种方法 first one：Louvain for community detection
#cluster的数量不是由自己规定的，算法会给出最优的类数，在这个题中有0到5，一共六类
partition = community_louvain.best_partition(g)
catagory = set([v for k,v in partition.items()])
for i in catagory:
    print('For',i,'cluster(s), the nodes are',[k for k,v in partition.items() if v == i])
#可视化看看结果visualize
pos = nx.spring_layout(g)
cmap = cm.get_cmap('summer', max(partition.values()) + 1)
nx.draw_networkx_nodes(g, pos, partition.keys(), node_size=50,cmap=cmap, label = True, node_color=list(partition.values()))
nx.draw_networkx_edges(g, pos, alpha=0.5)
nx.draw_networkx_labels(g, pos,font_size=8)
plt.show()



