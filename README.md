# HandwrittenDigitsAI
under development!

This project implements a neural network in C++ for recognizing handwritten digits, I converted my old code to object-oriented programming and added a backpropagation algorithm, reducing the training time from 11 years to 4.25 minutes.

The network is structured with three layers:
  - Layer 0: image values in scale (0.0,1.0)
  - Layer 1: 128 neurons.
  - Layer 2: (Output Layer): 10 neurons.

Functionality:
  - If no pre-trained model data is found in file, the program creates an untrained model with randomized weights.
  - The model data is stored and retrieved from the file.
  - The program can read an image file containing a 784-byte grayscale image (representing pixel brightness values from 0 to 255).
  - A digit is recognized by feeding the image data through the network using a sigmoid activation function.
  - The network can be trained using training images and labels provided in binary files.

Usage:
  - At startup, the program checks for an existing model.
  - Choose option 0 to recognize a digit from the input image or option 1 to train the network.
  
Note:
  - Ensure that the image file is formatted as a 784-byte file (28x28 pixels) in a left-to-right, top-to-bottom order.
  - Training files must be available for network training.

## License:
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

