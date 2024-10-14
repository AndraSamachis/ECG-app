from DatasetProcessor import DatasetProcessor
from Node import Node

def calculate_information_gain(dp, attribute):
    if dp.attribute_types[attribute] == 'numeric':
        split_point, information_gain = dp.find_best_split_point(attribute)
    else:
        entropy_before = dp.getEntropy()
        entropy_after = dp.calculate_conditional_entropy(attribute)
        information_gain = entropy_before - entropy_after
        split_point = None

    return information_gain, split_point

def ID3(branchName, attribList, dp):
    node = Node()
    node.branchName = branchName

    if len(dp.classLabels) == 1:
        node.name = dp.classLabels.pop()
        return node

    if not attribList:
        node.name = dp.getLabelWithMaxCount()
        return node

    best_attribute = None
    best_information_gain = -1
    best_split_point = None

    for attribute in attribList:
        information_gain, split_point = calculate_information_gain(dp, attribute)

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_attribute = attribute
            best_split_point = split_point

    node.name = best_attribute #cel mai bun atribut(cu cel mai mare ig)
    node.split_point = best_split_point

    if best_split_point is not None:
        left_subset = dp.ds[dp.ds[best_attribute] <= best_split_point]
        right_subset = dp.ds[dp.ds[best_attribute] > best_split_point]

        if not left_subset.empty:
            left_dp = DatasetProcessor(left_subset, dp.attribute_types)
            node.children.append(ID3(f'<= {best_split_point}', attribList, left_dp))
        if not right_subset.empty:
            right_dp = DatasetProcessor(right_subset, dp.attribute_types)
            node.children.append(ID3(f'> {best_split_point}', attribList, right_dp))
    else:
        Avalues = dp.getAttribValues(best_attribute)
        for val in Avalues:
            subset = dp.getSubset(best_attribute, val)
            if subset.empty:
                node.children.append(Node(name=dp.getLabelWithMaxCount(), branchName=val))
            else:
                newAttribList = [attr for attr in attribList if attr != best_attribute]
                node.children.append(ID3(val, newAttribList, DatasetProcessor(subset, dp.attribute_types)))

    return node


