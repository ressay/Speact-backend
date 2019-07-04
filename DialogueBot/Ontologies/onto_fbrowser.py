# Auto generated file #
from rdflib import Graph,URIRef
import os
import inspect


graph = Graph()
root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root += '/OwlFiles/'
f = root+'onto_browser.owl'
graph.parse(f,format='turtle')

# PREFIXES #

prefix1 = "http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#"


# CLASSES #

Directory = URIRef(prefix1+"Directory")
Copy_file = URIRef(prefix1+"Copy_file")
Dialogue_act = URIRef(prefix1+"Dialogue_act")
Act_parameter = URIRef(prefix1+"Act_parameter")
User = URIRef(prefix1+"User")
Parent_directory = URIRef(prefix1+"Parent_directory")
User_act = URIRef(prefix1+"User_act")
File = URIRef(prefix1+"File")
U_request = URIRef(prefix1+"U_request")
Delete_file = URIRef(prefix1+"Delete_file")
Open_file = URIRef(prefix1+"Open_file")
Old_name = URIRef(prefix1+"Old_name")
Dialogue = URIRef(prefix1+"Dialogue")
A_request = URIRef(prefix1+"A_request")
Slot = URIRef(prefix1+"Slot")
Move_file = URIRef(prefix1+"Move_file")
RegFile = URIRef(prefix1+"RegFile")
Create_file = URIRef(prefix1+"Create_file")
U_act_desire = URIRef(prefix1+"U_act_desire")
U_inform = URIRef(prefix1+"U_inform")
Change_directory = URIRef(prefix1+"Change_directory")
Rename_file = URIRef(prefix1+"Rename_file")
Agent = URIRef(prefix1+"Agent")
Agent_action = URIRef(prefix1+"Agent_action")
A_ask = URIRef(prefix1+"A_ask")
Desire = URIRef(prefix1+"Desire")
New_name = URIRef(prefix1+"New_name")
File_name = URIRef(prefix1+"File_name")
Agent_act = URIRef(prefix1+"Agent_act")
A_inform = URIRef(prefix1+"A_inform")


# RELATIONS #

# relation's domains: Agent
# relation's ranges: Agent_act
a_acted = URIRef(prefix1+"a_acted")
# relation's domains: Dialogue_act
# relation's ranges: Act_parameter
has_parameter = URIRef(prefix1+"has_parameter")
# relation's domains: User
# relation's ranges: Agent_action
confirm = URIRef(prefix1+"confirm")
# relation's domains: User
# relation's ranges: Desire
has_desire = URIRef(prefix1+"has_desire")
# relation's domains: User
# relation's ranges: Agent_action
deny = URIRef(prefix1+"deny")
# relation's domains: Change_directory
# relation's ranges: Directory
change_dir_to = URIRef(prefix1+"change_dir_to")
# relation's domains: Dialogue
# relation's ranges: Dialogue_act
contains_act = URIRef(prefix1+"contains_act")
# relation's domains: User
# relation's ranges: User_act
u_acted = URIRef(prefix1+"u_acted")
# relation's domains: Directory
# relation's ranges: File
contains_file = URIRef(prefix1+"contains_file")
# relation's domains: Dialogue_act
# relation's ranges: Dialogue_act
next_act = URIRef(prefix1+"next_act")
# relation's domains: File
# relation's ranges: Literal
has_name = URIRef(prefix1+"has_name")
# relation's domains: RegFile
# relation's ranges: string
has_extension = URIRef(prefix1+"has_extension")
