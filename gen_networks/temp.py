from utils import idf_to_str
import os

for i in range(6500, 10000):
    idf = idf_to_str(i)
    
    command = "netconvert --node-files=nod.xml --edge-files=edge_xmls/" + idf + ".xml --output-file=networks/" + idf + ".net.xml"
    os.system(command)

