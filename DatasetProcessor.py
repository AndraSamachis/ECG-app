import math


class DatasetProcessor:
    def __init__(self, ds, attribute_types):
        self.ds = ds
        self.className = ds.columns[-1]
        self.classLabels = set(ds[self.className])
        self.instanceCount = len(self.ds)

        self.attribute_types = attribute_types
        self.labelCount = {}
        self.labelProb = {}

        for label in self.classLabels:
            self.labelCount[label] = len(ds[ds[self.className] == label])
            self.labelProb[label] = self.labelCount[label] / self.instanceCount

    def getAttribValues(self, A):
        return list(set(self.ds[A]))

    def getSubset(self, A, Aval):
        return self.ds[self.ds[A] == Aval]

    def getLabelWithMaxCount(self):
        maxLabel = ''
        maxCount = 0
        for label in self.classLabels:
            if self.labelCount[label] > maxCount:
                maxLabel = label
                maxCount = self.labelCount[label]
        return maxLabel

    def getEntropy(self):
        entropy = 0
        for label in self.classLabels:
            p = self.labelProb[label]
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def calculate_conditional_entropy(self, attribute):
        total_instances = len(self.ds)
        attrib_values = self.getAttribValues(attribute)
        weighted_entropy = 0

        for value in attrib_values:
            subset = self.getSubset(attribute, value)
            subset_prob = len(subset) / total_instances
            subset_entropy = DatasetProcessor(subset, self.attribute_types).getEntropy()
            weighted_entropy += subset_prob * subset_entropy

        return weighted_entropy

    def find_best_split_point(self, attribute):
        if self.attribute_types[attribute] != 'numeric':
            raise ValueError(f"Attribute {attribute} is not numeric")

        sorted_data = self.ds.sort_values(by=[attribute])
        sorted_values = sorted_data[attribute].values
        sorted_classes = sorted_data[self.className].values

        best_split = None
        best_information_gain = -1
        entropy_before = self.getEntropy()

        for i in range(1, len(sorted_values)):
            if sorted_classes[i] != sorted_classes[i - 1]:
                split_point = (sorted_values[i] + sorted_values[i - 1]) / 2

                left_subset = sorted_data[sorted_data[attribute] <= split_point]
                right_subset = sorted_data[sorted_data[attribute] > split_point]

                left_entropy = DatasetProcessor(left_subset, self.attribute_types).getEntropy()
                right_entropy = DatasetProcessor(right_subset, self.attribute_types).getEntropy()

                left_prob = len(left_subset) / len(self.ds)
                right_prob = len(right_subset) / len(self.ds)

                entropy_after = left_prob * left_entropy + right_prob * right_entropy
                information_gain = entropy_before - entropy_after

                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_split = split_point

        return best_split, best_information_gain