# HandwrittenDigitsAI

This project implements a Feedforward Neural Network (FFNN) in C++. It is based on a previous codebase, now refactored into an object-oriented programming structure. The primary goal is educational, with an emphasis on simplicity by avoiding external libraries. The implementation does not rely on matrices; instead, calculations are performed iteratively, which is sufficient for certain use cases. The main file includes an example demonstrating its usage.

The program requires input files formatted exactly like the MNIST database, which is provided as an example. This includes both training and recognition data. Users can either run the main program directly to observe its functionality or use the NN class independently in other projects.

In the NN.hpp file, there is a #define directive that switches the program to debugging mode, where it displays information in the console.

### Build / Compilation Example
`g++ -std=c++17 main.cpp NN.cpp -o recognize -O2`
### Neural Network Layer Structure

The network structure is represented by three main components: `neuronLayers`, `weightLayers`, and `biasLayers`. These vectors contain the information necessary for both forward propagation and backpropagation in the network.

#### 1. `neuronLayers`
The `neuronLayers` vector holds the activations of the neurons at each layer of the network. This is a 2D vector, where the first index refers to the layer, and the second index corresponds to the neuron within that layer. In the context of backpropagation, it also holds the errors (gradients) during the learning process.

- `neuronLayers[layerIndex][neuronIndex]` gives the activation (or error) of the neuron at position `neuronIndex` in layer `layerIndex`.

For example:
- `neuronLayers[0]` would represent the input layer, where the values are the inputs to the network.
- `neuronLayers[1]` represents the first hidden layer, and so on.
- For backpropagation, once the error has been calculated, it will be stored in `neuronLayers[layerIndex]` for each layer.

#### 2. `weightLayers`
The `weightLayers` vector stores the weights connecting each layer to the next. It is a 2D vector, where the first index refers to the layer, and the second index corresponds to the weight connecting the neurons in the two layers. The number of weight layers is one less than the number of neuron layers because there are no weights for the input layer.

- `weightLayers[layerIndex]` contains all the weights between `neuronLayers[layerIndex]` and `neuronLayers[layerIndex + 1]`.
- `weightLayers[layerIndex][weightIndex]` represents the weight connecting the neuron at position `weightIndex` in `neuronLayers[layerIndex]` to the next layer (`neuronLayers[layerIndex + 1]`).

For example:
- `neuronLayers[0]` (input layer) is connected to `neuronLayers[1]` (first hidden layer) via `weightLayers[0]`.
- `weightLayers[0][0]` represents the connection between the first neuron of the input layer (`neuronLayers[0][0]`) and the first neuron of the hidden layer (`neuronLayers[1][0]`).
- `weightLayers[0][1]` represents the connection between the first neuron of the input layer and the second neuron of the hidden layer, and so on.

#### 3. `biasLayers`
The `biasLayers` vector holds the biases for each layer, except the input layer. Each element in `biasLayers` corresponds to the bias values associated with the neurons in the respective layer.

- `biasLayers[layerIndex][biasIndex]` represents the bias for the neuron at position `biasIndex` in `neuronLayers[layerIndex + 1]`.

For example:
- `biasLayers[0]` contains the biases for the neurons in `neuronLayers[1]` (the first hidden layer).
- `biasLayers[1]` contains the biases for the neurons in `neuronLayers[2]`, and so on.

Since the input layer doesn't have a bias, we start indexing biases from `biasLayers[0]`, which corresponds to `neuronLayers[1]`.

#### Explanation of Indexing
This structure, though seemingly non-intuitive, is designed to simplify the iterative calculations during the forward and backward passes. Instead of trying to use a more complex matrix structure like `weightLayers[layerIndex][neuronPreviousLayer][neuronNextLayer]`, which would introduce additional complexity, the current system allows for an efficient, iterative computation flow.

In summary:
- `neuronLayers[layerIndex]` contains the activations or errors for layer `layerIndex`.
- `weightLayers[layerIndex]` contains the weights between layer `layerIndex` and `layerIndex + 1`.
- `biasLayers[layerIndex]` contains the biases for `neuronLayers[layerIndex + 1]`.

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
