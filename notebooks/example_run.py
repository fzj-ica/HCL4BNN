from neural_network.binary_nn import NN
from datasets.sipm_dataset import SiPMDataset
from genetic_algorithm.algorithm import GeneticAlgorithm
from genetic_algorithm.utils import confusion_matrix, tuple_to_label


# GA parameters
n_samples = 128
n_frames = 50

# GA parameters
pop_size = 10
ngen = 5 

inp = SiPMDataset(n_samples = n_samples, n_frames = n_frames)

nn = NN((inp.n_samples, 32, 32, len(inp.LABLES)), input=inp, description="Good_vs_Ugly")

ga = GeneticAlgorithm(
    nn = nn, 
    pop_size = pop_size,
    ngen = ngen
)

pop, log, hof = ga.run()


print("Best fitness:", ga.nn.evaluate(hof[0])[0])
print("GA run complete.\n")


cm = confusion_matrix(tuple_to_label(nn.predictions), tuple_to_label(nn.targets), 3)
print("Confusion Matrix\n", cm)

