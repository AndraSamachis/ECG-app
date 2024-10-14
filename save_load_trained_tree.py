from Node import Node

def save_tree_to_file(node, file_path):
    with open(file_path, 'w') as file:
        write_node_to_file(node, file)

def write_node_to_file(node, file, level=0):
    indent = '  ' * level
    file.write(f"{indent}{node.name},{node.branchName},{node.split_point}\n")
    for child in node.children:
        write_node_to_file(child, file, level + 1)

def load_tree_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return read_node_from_lines(lines)

def read_node_from_lines(lines, level=0):
    if not lines:
        return None

    indent = '  ' * level
    line = lines.pop(0)
    name, branchName, split_point = line.strip().split(',')

    node = Node(name, branchName, split_point=split_point if split_point != 'None' else None)
    #cond de oprire pentru recursivitate
    while lines and lines[0].startswith(indent + '  '):
        child = read_node_from_lines(lines, level + 1)
        if child: #daca citim corect nodul copil
            node.children.append(child) #il adaugam la lista de copii ai lui node

    return node
