class Node:
    def __init__(self, name='root', branchName='', children=None, split_point=None):
        if children is None:
            children = []
        self.name = name
        self.branchName = branchName
        self.children = children
        self.split_point = split_point

    def __str__(self, level=0):
        ret = "\t" * level + repr(self) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return f"Node(name={self.name}, branchName={self.branchName}, split_point={self.split_point})"

