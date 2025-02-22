import csv, math, os, random
import xml.etree.ElementTree as ET
from config.run_simulation import run_simulation
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from sklearn.metrics import mean_absolute_error, r2_score

towns = ["dublin","antrim","craigavon","carlow","cavan","ennis","cork","derry","letterkenny","belfast","enniskillen","galway","tralee","naas","kilkenny","portlaoise","carrick","limerick","longford","dundalk","castlebar","navan","monaghan","tullamore","roscommon","sligo","clonmel","omagh","waterford","athlone","wexford","bray"]

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

town_coordinates = [
        ("dublin",3004,-2551),
        ("antrim",2968,-879),
        ("craigavon",2832,-1208),
        ("carlow",2567,-3222),
        ("cavan",2367,-1738),
        ("ennis",1162,-3162),
        ("cork",1497,-4368),
        ("derry",2293,-487),
        ("letterkenny",2028,-562),
        ("belfast",3180,-1008),
        ("enniskillen",2137,-1306),
        ("galway",1153,-2627),
        ("tralee",682,-3882),
        ("naas",2768,-2748),
        ("kilkenny",2368,-3411),
        ("portlaoise",2368,-3007),
        ("carrick",1808,-1787),
        ("limerick",1427,-3411),
        ("longford",2037,-2052),
        ("dundalk",2927,-1738),
        ("castlebar",982,-1922),
        ("navan",2768,-2167),
        ("monaghan",2518,-1408),
        ("tullamore",2272,-2643),
        ("roscommon",1728,-2142),
        ("sligo",1558,-1412),
        ("clonmel",2067,-3758),
        ("omagh",2292,-1007),
        ("waterford",2498,-3927),
        ("athlone",1938,-2407),
        ("wexford",2912,-3807),
        ("bray",3128,-2707)
]

def get_travel_time(simID, genome):
    edge_list = genome_to_edge_list(genome)
    try:
        sim = simulate(edge_list, simID)
    except:
        return 1.0
    normalized_time = normalize_sims(sim)
    return normalized_time

def edge_list_to_genome(edge_list):
    edges = [0] * (len(towns) ** 2)
    
    for [s,t,l,_,nL] in edge_list:
        assert s in towns and t in towns
        idx = get_idx(towns.index(s), towns.index(t))
        edges[idx] = int(nL)
    
    return edges

def genome_to_edge_list(genome):
    edge_list = []
    
    for i in range(len(genome)):
        if genome[i] > 0:
            source_idx, target_idx = get_source_target_idx(i)
            (_, x1, y1) = town_coordinates[source_idx]
            (_, x2, y2) = town_coordinates[target_idx]
            e_weight = math.dist([x1,y1], [x2, y2])
            edge_list.append([towns[source_idx], towns[target_idx], e_weight, 10, genome[i]])
    
    return edge_list

def simulate(edge_list, simID):
    node_path = "config/workspace/nod.xml"
    edge_path = "config/workspace/edg.xml"
    net_path = "config/workspace/current_network.net.xml"
    config_path = "config/workspace/base.sumo.cfg"

    tree = ET.Element("edges")
    i = 0
    
    for [f, t, l, s, n] in edge_list:
        edge_id = f"edge_{i}"
        ET.SubElement(tree,
                    "edge",
                    id=edge_id,
                    attrib={
                        "from": str(f),
                        "to": str(t),
                        "length": str(l),
                        "speed": str(s),
                        "numLanes": str(n)
                    })
        i += 1
    tree = ET.ElementTree(tree)
    tree.write(edge_path)
    
    command = "netconvert --node-files=" + node_path + " --edge-files=" + edge_path + " --output-file=" + net_path
    os.system(command)
    
    avg_distance, avg_speed = run_simulation(config_path)
    print("network " + simID + " with " + str(len(edge_list)) + " streets:"
        + "\n\taverage distance: " + str(avg_distance)
        + "\n\taverage speed:    " + str(avg_speed))
    
    time = avg_distance / avg_speed
    return time
    
def get_initial_population(interval=range(10000), directory="gen_networks/edge_csvs/"):
    population = []
    for i in interval:
        f = ''
        if i < 10:
            f = '0'
        if i < 100:
            f = f + '0'
        if i < 1000:
            f = f + '0'
        if i < 10000:
            f = f + '0'
        f = directory + f + str(i) + ".csv"
        
        with open(f, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            data.pop(0)
            
            genome = edge_list_to_genome(data)
            
            population.append(genome)
    
    return population

def normalize_sims(sim):
    return sim / (0.85 * 1092.2645637056744)

def normalize_street_lengths(length):
    return length / (0.75 * 665431.9038418838)

def get_source_target_idx(idx):
    source_idx = int(idx / len(towns))
    target_idx = idx % len(towns)
    return source_idx, target_idx

def get_idx(source_idx, target_idx):
    idx = len(towns) * source_idx + target_idx
    return idx

def get_street_length(genome):
    overall_length = 0
    for i in range(len(genome)):
        if genome[i] > 0:
            source_idx, target_idx = get_source_target_idx(i)
            (_, x1, y1) = town_coordinates[source_idx]
            (_, x2, y2) = town_coordinates[target_idx]
            e_weight = math.dist([x1,y1], [x2, y2])
            if source_idx > target_idx:
                if genome[get_idx(target_idx, source_idx)] > 0:
                    overall_length += genome[i] * 0.5 * e_weight
            else:
                overall_length += e_weight + (genome[i] -1) * 0.5 * e_weight
    
    normalized_length = normalize_street_lengths(overall_length)
    return normalized_length
            
def get_sims(f="training_data/data.csv"):
    travel_times = []
    
    with open(f, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        data.pop(0)
        
        i = 0
        for [network, avg_distance, avg_speed] in data:
            assert int(network) == i
            i += 1
            
            time = float(avg_distance) / float(avg_speed)
            normalized_time = normalize_sims(time)
            travel_times.append(normalized_time)
    
    return travel_times

def heal(genome):
    graph = nx.MultiDiGraph()
    
    for t in towns:
        graph.add_node(t)
    
    for i in range(len(genome)):
        if genome[i] > 0:
            source_idx = int(i / len(towns))
            target_idx = i % len(towns)
            graph.add_edge(towns[source_idx], towns[target_idx])
           
    if nx.is_strongly_connected(graph):
        return genome
        
    while not nx.is_strongly_connected(graph):
        components = list(nx.strongly_connected_components(graph))
    
        i = random.randint(0, len(components)-1)   
        comp1 = list(components[i])
    
        j = i
        while i == j:
            j = random.randint(0, len(components)-1)   
        
        comp2 = list(components[j])
    
        try:
            nx.shortest_path(graph, comp1[0], comp2[0])
    
            i = random.randint(0, len(comp2)-1)
            u2 = comp2[i]
            i = random.randint(0, len(comp1)-1)
            v2 = comp1[i]
            graph.add_edge(u2, v2)
            genome[towns.index(u2) * len(towns) + towns.index(v2)] = 1
    
        except:
            i = random.randint(0, len(comp1)-1)
            u1 = comp1[i]
            i = random.randint(0, len(comp2)-1)
            v1 = comp2[i]
            graph.add_edge(u1,v1)
            genome[towns.index(u1) * len(towns) + towns.index(v1)] = 1
        
    return genome

class GraphDataset(Dataset):
    def __init__(self, graph_list, y_values):
        self.graph_list = graph_list
        self.y_values = y_values

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        graph = self.graph_list[idx]
        y_value = self.y_values[idx]
        data = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=torch.tensor(y_value, dtype=torch.float))
        return data
    
    def append(self, graph, y):
        self.graph_list.append(graph)
        self.y_values.append(y)

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self, idx):
        return self.__len__()
        
    def get_graphs(self):
        return self.graph_list
    
    def get_y(self):
        return self.y_values

def get_emb_graph_data(genome):
    graph = nx.MultiDiGraph()
    
    town_to_idx = {name: idx for idx, (name, _, _) in enumerate(town_xy_list)}
    
    for name, x, y in town_xy_list:
        graph.add_node(town_to_idx[name], x=x, y=y)
    
    for i in range(len(genome)):
        for j in range(genome[i]):
            source_idx, target_idx = get_source_target_idx(i)
            (_, x1, y1) = town_coordinates[source_idx]
            (_, x2, y2) = town_coordinates[target_idx]
            weight = (math.dist([x1,y1], [x2,y2])) / 3961.7896208658026
            graph.add_edge(source_idx, target_idx, w=weight)
    
    return graph
            

def get_graph_data(genome):
    node_list = []
    for (t, x, y) in town_xy_list:
            node = [x, y]
            node_list.append(node)

    nodes = torch.tensor(node_list, dtype=torch.float)
    assert nodes.shape == torch.Size([32, 2])

    sources = []
    targets = []
    weights = []
    num_edges = 0
    
    for i in range(len(genome)):
        for j in range(genome[i]):
            source_idx, target_idx = get_source_target_idx(i)
            sources.append(source_idx)
            targets.append(target_idx)
            (_, x1, y1) = town_coordinates[source_idx]
            (_, x2, y2) = town_coordinates[target_idx]
            weight = math.dist([x1,y1], [x2, y2])
            weights.append([weight / 3961.7896208658026])
            num_edges += 1

    edges = torch.tensor([sources, targets], dtype=torch.long)
    assert edges.shape == torch.Size([2, num_edges])

    edge_attr = torch.tensor(weights, dtype=torch.float)
    assert edge_attr.shape == torch.Size([num_edges, 1])

    graph = Data(x=nodes, edge_index=edges, edge_weight=edge_attr, edge_attr=edge_attr)
    return graph

def get_model(model, train_dataset, val_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    num_epochs = 100
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    patience_counter = 0
    val_mae = 0
    val_r2 = 0
    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader)
        
        if epoch % 25 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')
        
        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter if improvement
        else:
            patience_counter += 1
        
        # Stop early if validation loss doesn't improve for `patience` epochs
        if patience_counter >= patience:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')
            print(f"Early stopping after {epoch} epochs due to no improvement in validation loss.")
            break
    
    return model, val_mae, val_r2

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            output = model(data)
            loss = F.mse_loss(output, data.y.view(-1, 1))
            total_loss += loss.item()
            all_outputs.append(output.view(-1).cpu())
            all_targets.append(data.y.cpu())
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(all_targets.numpy(), all_outputs.numpy())
    
    # Calculate R2 Score
    r2 = r2_score(all_targets.numpy(), all_outputs.numpy())
    
    return total_loss / len(loader), mae, r2

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        
        # Node feature input size: 2 -> Hidden size: 32
        self.node_emb = torch.nn.Linear(2, 32)
        
        # Edge feature input size: 1 -> Hidden size: 32
        self.edge_emb = torch.nn.Linear(1, 32)
        
        # Graph convolution layers
        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 64)
        
        # Fully connected layer for graph-level output
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Embed node features
        x = self.node_emb(x)
        x = F.relu(x)
        
        # Embed edge features
        edge_attr = self.edge_emb(edge_attr)
        edge_attr = F.relu(edge_attr)

        # Apply graph convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Pooling (aggregate node features to a graph-level representation)
        x = global_mean_pool(x, data.batch)  # Assumes we are batching graphs
        
        # Fully connected layers to predict graph-level output
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Output a single scalar per graph

        return x
