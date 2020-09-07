from numpy import exp, array, random, dot

training_sets_inputs = array([[0, 0, 1], [1, 1, 1], [0, 1, 1]])
training_sets_outputs = array([[0, 1, 1, 0]]).T

random.seed(1)

synaptic_weights = 2 * random((3, 1)) - 1
for iteration in xrange(10000):
	output =  1 / (1 + exp(-(dot(training - set_inputs, synaptic_weights))))
	synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output)
	
pred = 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
