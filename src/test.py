import numpy as np
import matplotlib.pyplot as plt
from net2 import neuralNet

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.01

# neural net instances
nnet = neuralNet(input_nodes,hidden_nodes, output_nodes, learning_rate)
nnet.load('model_trained.npz')

test_data = open('mnist_test.csv', 'r')
test_data_list = test_data.readlines()
test_data.close()

all_values = test_data_list[5].split(',')
scaled_input = (np.asarray(all_values[1:], dtype=int) / 255.0 * 0.99) + 0.01


score = []
for i in range(len(test_data_list)):
      all_values = test_data_list[i].split(',')
    # Scale input values
      scaled_input = (np.asarray(all_values[1:], dtype=int) / 255.0 * 0.99) + 0.01
      reg = nnet.quary(scaled_input)
      regP = np.argmax(reg)
      label = all_values[0]
      print(f"label = {label}, Recognise = {regP}")
      if (regP == int(label)):
        print(f'Correct!')
        score.append(1)
      else:
        print(f'wrong!')
        score.append(0)
score_array = np.array(score)
print("performance =", score_array.sum() / score_array.size)
