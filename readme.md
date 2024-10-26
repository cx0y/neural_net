# How ANN (Articial Neural Network) Works? 
Well, we need some mathematics here, so keep with me, ANN has some conceps, like Activation Functions, Layers, Weights, Back Propagration, Optimaization
I'll try to cover the concepts, easy way (how I got) 

---
1. **Artifical Neurons:** Artifical Neurons are copied version of How original neurons works (partially understood), biological neurons takes some input from it's dendrites, and add them,
    together and put them some kind of step function (activation function here), here is a catch if, combined input is not enough, neuron don't fire, and if combined input enough then neuron fire
    and neuron just don't fire, it's learns also, what does that mean? well it's means, a neuron have 100 connections with other neurons, but which one are most important it's recgnise during learning process
    and artifical neuron just mimic this things
---
2. **Activation function:** A function (mathematical here) which controls the firing of Artificial neuron, this function take combined inputs of the neural connectios and determines neurons state
    like if `output > 1` then fired, if not then neuron didn't fired,
    <br>
    <br>
    *well how an activation function looks like?* [more info](https://en.wikipedia.org/wiki/Activation_function)
     well, hare is an example of activation function, `sigmoid` function which I used for my this project
   
     <div align="center">
       $$sigmoid(x) = \frac{1}{1 + e^{-x}}$$
     </div>
     <br>
     
     ```py
     # code of activation function
     def actFunc(self, x):
        from numpy import exp
        return (1/(1 + exp(-x)))
     ```
     

      
     