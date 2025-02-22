# HandwrittenDigitsAI
(under development)
This project implements a neural network in C++ for recognizing handwritten digits.
The network is structured with three layers:
  - Layer 1: 784 bias values and 614656 weight values.
  - Layer 2: 784 bias values and 614656 weight values.
  - Layer 3 (Output Layer): 10 bias values and 7840 weight values.

Functionality:
  - If no pre-trained model data is found in "model_data", the program creates an untrained model with randomized weights.
  - The model data is stored and retrieved from the file "model_data".
  - The program can read an image file ("roz") containing a 784-byte grayscale image (representing pixel brightness values from 0 to 255).
  - A digit is recognized by feeding the image data through the network using a sigmoid activation function.
  - The network can be trained using training images and labels provided in binary files "img" and "lab".

Usage:
  - At startup, the program checks for an existing model.
  - Choose option 0 to recognize a digit from the input image or option 1 to train the network.
  
Note:
  - Ensure that the image file "roz" is formatted as a 784-byte file (28x28 pixels) in a left-to-right, top-to-bottom order.
  - Training files "img" and "lab" must be available for network training.
