import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

D = [3.3, 3.5, 3.1, 1.8, 3.0, 0.74, 2.5, 2.4, 1.6, 2.1, 2.4, 1.3, 1.7, 0.19]

def mle_predictor(x, data):
	pass
	# TODO: Implement this based on 1.b

def map_predictor(x, data):
	pass
	# TODO: Implement this based on 1.d

def posterior_predictive(x, data):
	pass
	# TODO: Implement this based on 1.e

def graph():
	# After you implement the first three funcitons, this will graph your pdfs
	fig, ax = plt.subplots(nrows=5, ncols=3)
	x = np.arange(-8, 8, 0.1)
	data_idx = 1
	for row in ax:
		for col in row:
			trimmed_data = D[:data_idx]
			col.plot(x, mle_predictor(x, trimmed_data), alpha=0.5)
			col.plot(x, map_predictor(x, trimmed_data), alpha=0.5)
			col.plot(x, posterior_predictive(x, trimmed_data), alpha=0.5)
			col.set_title(f"{data_idx} points")
			data_idx += 1
	fig.legend(["MLE", "MAP", "Posterior Predictive"])
	fig.subplots_adjust(hspace=0.75) # Adjust this if your title are overlapping with graphs
	plt.show()

def marginal_likelihood(tau_squared):
	pass
	# If you want, you can use this function to calculate the answers for 5 and 6

if __name__ == "__main__":
	graph()
