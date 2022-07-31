from EnsembleLearning import *
from EnsembleLearning.Ensemble import *

raw_data = DataPrepare()

raw_data.data_raw('data')

raw_data.separete_Xy()

print(raw_data.y)

