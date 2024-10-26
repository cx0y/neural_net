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
     example, if a neuron has 3 inputs, `a1` , `a2`, `a3` , then the `total_input = a1 + a2 + a3` , and the output from the neuron is, `output = activationFunction(total_input)` , the total
     output determines the activation of neuron 

---

3. **Weights:** weights, what I understands (intuitively) is weights determines the strengths between neuron connections, wait? what? well, as I say, if a neuron has 100 connections,       when it’s learning it’s doesn’t needs all connections to perform that task, some connections are more important (have greater impact on neuron activation), so how we filter out those connections, we just put some weights between inputs, which connection have more weight, have greater impact in neuron activation, but question remains how we do that? easy
<br>

back to the example, if a neuron has 3 inputs (`a1`, `a2`, `a3`), the we put 3 weights between them, `w1`, `w2`, `w3`, then the weighted total input is, `weighted_total_input = w1*a1 + w2*a2 + w3*a3` 

and again we put it in activation function, `output = activationFunction(weighted_total_input)`

     