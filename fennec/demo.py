import numpy as np

from evaluate import Experiment
from metrics import MetricEval

import methods
from methods import PARC, HScore, kNN, NegativeCrossEntropy, \
LEEP, RawHSIC, DDS, RSA, KITE_HSIC, LogME, NLEEP, GBC, TransRate, SFDA, kernal_FDA, LDA


# Set up the methods to use.
# To define your own method, inherit methods.TransferabilityMethod. See the methods in methods.py for more details.
my_methods = {
	'RSA': RSA(reference_architecture='resnet50', n_dims=32),
	'DDS': DDS(reference_architecture='resnet50', n_dims=32),
	
	# 'NCE' : NegativeCrossEntropy(),	
	'LEEP' : LEEP(n_dims=32),


	'H-Score': HScore(n_dims=32),
	'LogME': LogME( n_dims=32),
	'TransRate' : TransRate(n_dims=32),
	'PARC': PARC(n_dims=32),
	'GBC' : GBC(gaussian_type='spherical', n_dims=32),# 'diagonal' 'spherical'
	'SFDA' : SFDA(n_dims=32),


}


experiment = Experiment(my_methods, name='test', append=False, budget=500, full=False) 
#experiment = Experiment(my_methods, name='test', append=False, budget=1000) # Set up an experiment with those methods named "test".
                                                               # Append=True skips evaluations that already happend. Setting it to False will overwrite.

experiment.run()                                               # Run the experiment and save results to ./results/{name}.csv

metric = MetricEval(experiment.out_file)                       # Load the experiment file we just created with the default oracle
#metric.add_plasticity()                                        # Adds the "capacity to learn" heuristic defined in the paper
mean, variance, _all = metric.aggregate()                      # Compute metrics and aggregate them
print(mean)
# for method in mean:
# 	avg_time = sum(timing[method]) / len(timing[method]) if method in timing else 0
# 	print(f'{method:20s}: {mean[method]:6.2f}% +/- {np.sqrt(variance[method]):4.2f} ({avg_time*1000:.1f} ms +/- {np.std(timing[method])*1000:.1f})')

