# HandwrittenDigitsAI

This project implements a Feedforward Neural Network (FFNN) in C++. It is based on a previous codebase, now refactored into an object-oriented programming structure. The primary goal is educational, with an emphasis on simplicity by avoiding external libraries. The implementation does not rely on matrices; instead, calculations are performed iteratively, which is sufficient for certain use cases. The main file includes an example demonstrating its usage.

The program requires input files formatted exactly like the MNIST database, which is provided as an example. This includes both training and recognition data. Users can either run the main program directly to observe its functionality or use the NN class independently in other projects.

In the NN.hpp file, there is a #define directive that switches the program to debugging mode, where it displays information in the console.

## Mathematical Concepts Employed:

- Sigmoid Activation Function
- Xavier Initialization
- Mean Squared Error (MSE) Cost Function
- Backpropagation Algorithm
- Stochastic Gradient Descent (Online Learning)

## Backpropagation Algorithm (Training):

### 1. Forward Pass
The forward pass computes activation values for each layer:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \sigma(z^{(l)})
$$

Where:

- $W^{(l)}$ – **Weight matrix** for layer $l$  
- $b^{(l)}$ – **Bias vector** for layer $l$  
- $a^{(l)}$ – **Activation vector** for layer $l$  
- $z^{(l)}$ – **Weighted sum before activation** for layer $l$
- $\sigma$ – **Activation function**  

### 2. Backpropagation (Gradient Calculation)

#### For the Output Layer
Compute the error for the output layer:

$$
\delta^{(out)} = 2(a^{(out)} - y) \odot \sigma'(z^{(out)})
$$

Where:

- $\delta^{(out)}$ – **Error** for the output layer  
- $\sigma'$ – **Derivative of the activation function**  
- $\odot$ – **Hadamard product (element-wise multiplication)**
- $y$ - **Desired output**

#### For the Hidden Layer
Compute the error for the hidden layer:

$$
\delta^{(l)} = (W^{(l+1)T} \delta^{(l+1)}) \odot \sigma'(z^{(l)})
$$

Where:

- $W^{(l)T}$ – **Transposed weight matrix**

### Weight and Bias Gradients:
$$
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T
$$

$$
\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}
$$

Where:

- $\frac{\partial L}{\partial W^{(l)}}$ – **Gradient with respect to the weights**  
- $\frac{\partial L}{\partial b^{(l)}}$ – **Gradient with respect to the biases**

### 3. Weight and Bias Updates (Gradient Descent – Optimization)
$$
W^{(l)} = W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}
$$

$$
b^{(l)} = b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}
$$

Where:

- $\eta$ – **Learning rate (scalar value)**  


## License:
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
