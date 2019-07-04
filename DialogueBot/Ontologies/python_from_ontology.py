import inspect
import os

from rdflib import Graph,RDFS,RDF,OWL
from itertools import chain

owl_class = OWL.term('Class')
owl_obj_property = OWL.term('ObjectProperty')
owl_data_property = OWL.term('DatatypeProperty')
rdf_type = RDF.term('type')
rdfs_domain = RDFS.term('domain')
rdfs_range = RDFS.term('range')
rdfs_subClassOf = RDFS.term('subClassOf')

def create_onto_py_URIs(file,form='turtle',add_imports=True):
    g = Graph()
    root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    root += '/OwlFiles/'
    g.parse(root + file,format=form)
    file_content = ''
    if add_imports:
        file_content = "# Auto generated file #\n" \
                        "from rdflib import Graph,URIRef\n" \
                        "import os\n" \
                        "import inspect\n\n\n" \

    file_content += "graph = Graph()\n" \
                    "root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))\n" \
                    "root += '/OwlFiles/'\n" \
                    "f = root+'"+file+"'\n" \
                    "graph.parse(f,format='"+form+"')\n\n"
    prefixes = {}
    classes = {}
    relations = {}
    comments = {}
    count = 1
    for s,p,o in g.triples((None,rdf_type,owl_class)):
        prefix,c = str(s).split('#')
        if prefix not in prefixes:
            prefixes[prefix] = count
            count += 1
        classes[c] = prefixes[prefix]

    for s,p,o in chain(g.triples((None,rdf_type,owl_obj_property)),
                       g.triples((None,rdf_type,owl_data_property))):
        prefix,r = str(s).split('#')
        if prefix not in prefixes:
            prefixes[prefix] = count
            count += 1
        relations[r] = prefixes[prefix]
        domains = " ".join([str(ob).split('#')[1] for su, pr, ob in g.triples((s, rdfs_domain, None))])
        ranges = " ".join([str(ob).split('#')[1] for su, pr, ob in g.triples((s, rdfs_range, None))])
        comments[r] = (domains,ranges)



    print(classes)
    print(prefixes)
    file_content += '# PREFIXES #\n\n'
    for prefix in prefixes:
        file_content += 'prefix'+str(prefixes[prefix])+' = "'+prefix+'#"\n'
    file_content += '\n\n# CLASSES #\n\n'
    for c in classes:
        file_content += c + ' = URIRef(prefix'+str(classes[c])+'+"'+c+'")\n'
    file_content += '\n\n# RELATIONS #\n\n'
    for r in relations:
        domains,ranges = comments[r]
        file_content += "# relation's domains: "+domains+"\n# relation's ranges: "+ranges+"\n"
        file_content += r + ' = URIRef(prefix'+str(relations[r])+'+"'+r+'")\n'
    return file_content

if __name__ == '__main__':
    content = create_onto_py_URIs('onto_browser.owl')
    f = open('onto_fbrowser.py','w')
    f.write(content)
    f.close()