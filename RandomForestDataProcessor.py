import random
from DatasetProcessor import DatasetProcessor

class RandomForestDataProcessor(DatasetProcessor):
    def __init__(self, ds, attribute_types):
        super().__init__(ds, attribute_types)
        self.attributes = list(ds.columns[:-1])

    #alegem un subset aleatoriu din dataset-ul original
    def generateRandomSubset(self):
        #alegem un nr aleatoriu de instante (nr intre 2si nr instante -1)
        count = random.randint(2, self.instanceCount - 1)
        #selectam aleatoriu count instante din dataset, fara a le repeta
        indices = random.sample(range(self.instanceCount), count)

        #returneaza subsetul cu instantele alese
        return self.ds.iloc[indices, :]

    #returneaza o lista care contine toate atributele datasetului
    def getSelectedAttribList(self):
        return [attr for attr in self.attributes]

