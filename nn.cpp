#include <vector>
#include <iostream>
#include <cmath>
#include <random>  // for random number generation

// neuron class represents a single neuron in the neural network
class Neuron {
public:
    double value;  // stores the output value of the neuron
    std::vector<double> weights;  // stores the weights for each input connection
    double bias;  // stores the bias of the neuron

    // constructor: initializes a neuron with random weights and bias
    Neuron(int num_inputs) {
        // set up random number generation
        std::random_device rd;  // used to obtain a seed for the random number engine
        std::mt19937 gen(rd());  // standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-1.0, 1.0);  // defines the range of random numbers
        
        // initialize weights with random values
        weights.resize(num_inputs);
        for (auto& weight : weights) {
            weight = dis(gen);  // assign a random value between -1 and 1
        }
        bias = dis(gen);  // initialize bias with a random value
    }

    // activation function: processes inputs and produces an output
    double activate(const std::vector<double>& inputs) {
        double sum = bias;
        // calculate the weighted sum of inputs
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        value = sigmoid(sum);  // apply the sigmoid activation function
        return value;
    }

    // sigmoid activation function
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // derivative of the sigmoid function (used in backpropagation)
    static double sigmoidDerivative(double x) {
        double sig = sigmoid(x);
        return sig * (1.0 - sig);
    }
};

// layer class represents a layer of neurons in the neural network
class Layer {
public:
    std::vector<Neuron> neurons;  // stores the neurons in this layer

    // constructor: creates a layer with the specified number of neurons
    Layer(int num_neurons, int num_inputs) {
        neurons.reserve(num_neurons);  // to avoid vector having to reallocate memory for itself as elements are added
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(num_inputs);  // create each neuron with the given number of inputs
        }
    }

    // processes inputs through this layer and returns the outputs
    std::vector<double> feedForward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        outputs.reserve(neurons.size());
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs));  // activate each neuron and store its output
        }
        return outputs;
    }
};

// neuralnetwork class represents the entire neural network
class NeuralNetwork {
public:
    std::vector<Layer> layers;  // stores all layers in the network

    // constructor: creates a neural network with the specified topology
    NeuralNetwork(const std::vector<int>& topology) {
        // create layers based on the topology
        for (size_t i = 1; i < topology.size(); ++i) {
            layers.emplace_back(topology[i], topology[i-1]);
        }
    }

    // processes inputs through the entire network and returns the final output
    std::vector<double> feedForward(const std::vector<double>& inputs) {
        std::vector<double> current_inputs = inputs;
        // pass inputs through each layer in turn
        for (auto& layer : layers) {
            current_inputs = layer.feedForward(current_inputs);
        }
        return current_inputs;  // the final layer's outputs are the network's outputs
    }

    // checking if everything is set up okay
    void displayNN() {
        
        for (Layer& layer : layers) {
            std::cout << "\n\n" << "next layer\n";

            for (Neuron& neuron : layer.neurons) {
                std::cout << "neuron with weights {";
                
                for (auto i : neuron.weights) {
                    std::cout << i;
                    std::cout << " ";
                }

                std::cout << "}\n";
            }
        }
    }
};

int main() {
    std::vector<int> topology = {2, 3, 1};
    NeuralNetwork nn = NeuralNetwork(topology);
    nn.displayNN();
}