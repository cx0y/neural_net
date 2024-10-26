import numpy as np
import matplotlib.pyplot as plt
from net2 import neuralNet

# with open("mnist_train.csv", 'r') as data_file:
#     data_list = data_file.readlines()



# all_values = data_list[2].strip().split(',')
# #image_array = np.asarray([int(value) for value in all_values[1:] if value.isdigit()]).reshape((28, 28))
# image_array = np.asarray(all_values[1:], dtype=int).reshape((28, 28))
# scaled_input = (np.asarray(all_values[1:], dtype=int) / 255.0 * 0.99) + 0.01

# #print(scaled_input)

# #plt.imshow(image_array, cmap='Greys', interpolation='none')
# #plt.show()

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.5

# neural net instances
nnet = neuralNet(input_nodes,hidden_nodes, output_nodes, learning_rate)

#loading traning data
traning_data_file = open("mnist_train.csv", 'r')
traning_data_list = traning_data_file.readlines()
traning_data_file.close()
print(len(traning_data_list))

for i in range(len(traning_data_list)):
      all_values = traning_data_list[i].strip().split(',')
    # Scale input values
      scaled_input = (np.asarray(all_values[1:], dtype=int) / 255.0 * 0.99) + 0.01
    
      #Target output
      targets = np.zeros(output_nodes) + 0.01
      targets[int(all_values[0])] = 0.99  # Use label at all_values[0]

#     # Train the network
      nnet.train(scaled_input, targets)

# # Save trained model
nnet.save('model_trained.npz')
