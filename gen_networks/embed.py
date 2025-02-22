from karateclub import LDP
import numpy as np
import pandas as pd
import networkx as nx
import sys, csv, math

model = LDP()
filename = "LDP_embedding.py"

town_xy_list = [
		("dublin", 0.9295436349079264, 0.4681783045606802), 
		("antrim", 0.9151321056845476, 0.8989951043545478), 
		("craigavon", 0.8606885508406725, 0.8142231383664004), 
		("carlow", 0.754603682946357, 0.2952847204328781), 
		("cavan", 0.6745396317053642, 0.6776603968049472), 
		("ennis", 0.1921537229783827, 0.3107446534398351), 
		("cork", 0.3262610088070456, 0.0), 
		("derry", 0.644915932746197, 1.0), 
		("letterkenny", 0.5388310648518815, 0.9806750837413037), 
		("belfast", 1.0, 0.8657562483895903), 
		("enniskillen", 0.5824659727782225, 0.7889719144550373), 
		("galway", 0.18855084067253802, 0.4485957227518681), 
		("tralee", 0.0, 0.12522545735635146), 
		("naas", 0.8350680544435548, 0.4174181911878382), 
		("kilkenny", 0.6749399519615693, 0.24658593146096366), 
		("portlaoise", 0.6749399519615693, 0.35068281370780724), 
		("carrick", 0.45076060848678945, 0.6650347848492657), 
		("limerick", 0.29823859087269816, 0.24658593146096366), 
		("longford", 0.5424339471577262, 0.596753414068539), 
		("dundalk", 0.8987189751801441, 0.6776603968049472), 
		("castlebar", 0.1200960768614892, 0.6302499355836124), 
		("navan", 0.8350680544435548, 0.5671218758052048), 
		("monaghan", 0.7349879903923139, 0.7626900283432105), 
		("tullamore", 0.6365092073658927, 0.4444730739500129), 
		("roscommon", 0.4187349879903923, 0.5735635145581036), 
		("sligo", 0.3506805444355484, 0.7616593661427468), 
		("clonmel", 0.5544435548438751, 0.1571759855707292), 
		("omagh", 0.644515612489992, 0.8660139139397063), 
		("waterford", 0.7269815852682145, 0.11363050760113373), 
		("athlone", 0.5028022417934348, 0.505282143777377), 
		("wexford", 0.8927141713370697, 0.14455037361504766), 
		("bray", 0.9791833466773419, 0.4279824787425921)
]

def get_edge_weight(s, t):
    x1,x2,y1,y2 = 0
    for (n, x, y) in town_xy_list:
        if n == s:
            x1 = x
            y1 = y
        elif n == t:
            x2 = x
            y2 = y
    
    weight = math.dist([x1, y1], [x2, y2])
    return weight

def get_graph_filename(i):
    s = str(i)
    if i < 10:
        s = '0' + s
    if i < 100:
        s = '0' + s
    if i < 1000:
        s = '0' + s
    if i < 10000:
        s = '0' + s
    
    s = "edge_csvs/" + s + ".csv"
    return s

def get_graph(i):
    graph = nx.MultiDiGraph()
    
    for (name, x, y) in town_xy_list:
        graph.add_node(name, x=x, y=y)

    graph_filename = get_graph_filename(i)
    with open(graph_filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        data.pop(0)
        
        for (s, t, _, _, l) in data:
            for j in range(l):
                weight = get_edge_weight(s,t)
                graph.add_edge(s, t, w=weight)
    
    return graph
    

graphs = []
for i in range(10000):
    graphs.append(get_graph(i))
    
model.fit(graphs)
embedding = model.get_embedding()

print(len(embedding))
print(len(embedding[0]))
print(embedding[0])

fp = open(filename, 'w')
fp.write("embeddings = " + str(embedding))
