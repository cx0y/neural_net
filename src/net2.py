
from numpy import dot, array, random, shape, exp, transpose, savez, load 

class neuralNet:
    
    def __init__(self, input_node, hidden_node, output_node, learningRate):
        'initialization of neural network'
        self.iNode = input_node 
        self.hNode = hidden_node
        self.oNode = output_node
        self.lr = learningRate
        #hidden_layer 
        self.w_in = random.normal(0.0, pow(self.hNode, -0.5), (self.hNode, self.iNode))
        self.w_out = random.normal(0.0, pow(self.oNode, -0.5), (self.oNode, self.hNode))
        pass
    
    def actFunc(self, x):
        return (1/(1 + exp(-x)))
        pass 

    def train(self, inputData, targetData):
        inputs = array(inputData, ndmin=2).T
        targets = array(targetData, ndmin=2).T 

        #input_layer
        inputs = array(inputData, ndmin=2).T

        # #hidden_layer_work (1 for input, 2 for output)
        h1_1 = dot(self.w_in, inputs)
        h1_2 = self.actFunc(h1_1)

        #final_output_layer
        f1_1 = dot(self.w_out, h1_2)
        outputs = self.actFunc(f1_1)
        #Error
        output_err = targets - outputs #output_error 
        h1_2_err = dot(self.w_out.T, output_err) #previous_hidden_output_err

        #gredient_descend
        self.w_out += self.lr * dot((output_err * outputs * (1 - outputs)), transpose(h1_2))
        self.w_in += self.lr * dot((h1_2_err * h1_2 * (1 - h1_2)) , transpose(inputs))
        print(output_err)
        pass 

    def quary(self, inputData):
        #input_layer
        inputs = array(inputData, ndmin=2).T

        # #hidden_layer_work (1 for input, 2 for output)
        h1_1 = dot(self.w_in, inputs)
        h1_2 = self.actFunc(h1_1)

        #final_output_layer
        f1_1 = dot(self.w_out, h1_2)
        outputs = self.actFunc(f1_1)

        #print(inputs.shape, outputs.shape)
        return outputs
        pass
    
    def save(self, filename="model_weights.npz"):
        savez(filename, w_in=self.w_in, w_out=self.w_out)

    def load(self, filename="model_weights.npz"):
        # Load weights and biases from an .npz file
        data = load(filename)
        self.w_in = data['w_in']
        self.w_out = data['w_out']
        print(f"Model loaded from {filename}")