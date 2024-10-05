#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>

// matrix class for handling matrix operations
class Matrix {
public:
    std::vector<std::vector<double>> data;  // 2D vector to store matrix data
    size_t rows, cols;  // dimensions of the matrix

    // constructor: initialize matrix with given dimensions
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<double>(cols, 0.0));  // resise the data vector to have 'rows' elements with a vector as the value
    }

    // initialize matrix with random values between -1 and 1
    void uniform_initialise() {
        std::random_device rd;  // obtain a random number from hardware
        std::mt19937 gen(rd());  // seed the generator
        std::uniform_real_distribution<> dis(-1.0, 1.0);  // define the range
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);  // generate random number
            }
        }
    }


    // initialize matrix with random values between -1 and 1
    void zero_initialise() {
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = 0;  // generate random number
            }
        }
    }

    // for sigmoid activation
    void xavier_initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = sqrt(6.0 / (rows + cols));
        std::uniform_real_distribution<> dis(-limit, limit);
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);
            }
        }
    }


    // for ReLU activation
    void he_initialise() {
        std::random_device rd;
        std::mt19937 gen(rd());
        double std_dev = sqrt(2.0 / cols);  
        std::normal_distribution<> dis(0, std_dev);
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);
            }
        }
    }

    // matrix multiplication overloading the * operator 
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
        std::cerr << "Attempted to multiply matrices of incompatible dimensions: "
                  << "(" << rows << "x" << cols << ") * (" 
                  << other.rows << "x" << other.cols << ")" << std::endl;
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
                for (size_t k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // element-wise addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // element-wise subtraction 
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    // scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }

    // transpose 
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    // hadamard producht
    Matrix hadamard(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    // apply a function element-wise to the matrix
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result; 
    }
};

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Sigmoid derivative
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// layer class representing a single layer in the neural network
class Layer {
public:
    Matrix weights;
    Matrix bias;
    Matrix output;
    Matrix input;  // store input for backpropagation
    Matrix z;      // store weighted input for backpropagation
    // storing 'vectors' as matrix types to reuse matmul code (mathematicians close your eyes).

    // constructor: initialize weights and biases with random values
    Layer(size_t input_size, size_t output_size) 
        : weights(output_size, input_size),
          bias(output_size, 1),
          output(output_size, 1),
          input(input_size, 1),
          z(output_size, 1) {
        weights.xavier_initialize();
        bias.zero_initialise();
    }

    // perform feedforward operation for this layer
    Matrix feedforward(const Matrix& input) {
        this->input = input;  // Store input
        z = weights * input + bias;  // Store pre-activation
        output = z.apply(sigmoid);
        return output;
    }
};

// neural network class
class NeuralNetwork {
public:
    std::vector<Layer> layers;  // vector to store all layers

    // constructor: create layers based on the given topology
    NeuralNetwork(const std::vector<size_t>& topology) {
        for (size_t i = 1; i < topology.size(); i++) {
            layers.emplace_back(topology[i-1], topology[i]);
        }
    }

    // perform feedforward operation through all layers
    Matrix feedforward(const Matrix& input) {
        Matrix current = input;
        for (auto& layer : layers) {
            current = layer.feedforward(current);
        }
        return current;
    }

    // updates weights and biases for a single training example
    void backpropagate(const Matrix& input, const Matrix& target, double learning_rate) {
        // Forward pass
        std::vector<Matrix> activations = {input};
        for (auto& layer : layers) {
            activations.push_back(layer.feedforward(activations.back()));
        }

        if (activations.back().data[0][0] == 0.0) {
            for (size_t i; i < layers.size(); ++i) {
                std::cout << "l" << i << ": ";
                for (auto& neuron : layers[i].output.data) {
                    std::cout << neuron[0] << ",";
                }
                std::cout << "\n";
            }
        }

        // Backward pass
        int num_layers = layers.size();
        std::vector<Matrix> deltas;
        deltas.reserve(num_layers);

        // 1. Output layer error (δ^L = ∇_a C ⊙ σ'(z^L))
        Matrix output_error = activations.back() - target;
        Matrix output_delta = output_error.hadamard(layers.back().z.apply(sigmoid_derivative));
        deltas.push_back(output_delta);

        // 2. Hidden layer errors (δ^l = ((w^(l+1))^T δ^(l+1)) ⊙ σ'(z^l))
        for (int l = num_layers - 2; l >= 0; l--) {
            Matrix delta = (layers[l+1].weights.transpose() * deltas.back()).hadamard(layers[l].z.apply(sigmoid_derivative));
            deltas.push_back(delta);
        }

        // Reverse deltas to match layer order
        std::reverse(deltas.begin(), deltas.end());

        // 3 & 4. Update weights and biases
        for (int l = 0; l < num_layers; l++) {
            // ∂C/∂b^l = δ^l
            layers[l].bias = layers[l].bias - (deltas[l] * learning_rate);

            // ∂C/∂w^l = δ^l (a^(l-1))^T
            Matrix weight_gradient = deltas[l] * activations[l].transpose();
            layers[l].weights = layers[l].weights - (weight_gradient * learning_rate);
        }
    }
};


// Mean Squared Error (MSE) loss function
double mse_loss(const Matrix& predicted, const Matrix& target) {
    if (predicted.rows != target.rows || predicted.cols != target.cols) {
        throw std::invalid_argument("Dimensions of predicted and target matrices don't match");
    }

    double sum = 0.0;
    for (size_t i = 0; i < predicted.rows; ++i) {
        for (size_t j = 0; j < predicted.cols; ++j) {
            double diff = predicted.data[i][j] - target.data[i][j];
            sum += diff * diff;
        }
    }
    return sum / (predicted.rows * predicted.cols);
}


int main() {
    NeuralNetwork nn({5, 3, 3, 1});
    
    Matrix input(5, 1);
    input.data = {{1}, {0.5}, {0.9}, {-0.5}, {0.01}};  // example input
    Matrix target(1, 1);
    target.data = {{0.7}};  // example target

    // Training loop
    for (int epoch = 0; epoch < 1000; epoch++) {
        nn.backpropagate(input, target, 0.1);
        
        if (epoch % 100 == 0) {
            Matrix output = nn.feedforward(input);
            double loss = mse_loss(output, target);
            std::cout << "Epoch " << epoch << ", Loss: " << loss << "\n";
        }
    }

    // Final prediction
    Matrix final_output = nn.feedforward(input);
    std::cout << "Final Output: " << final_output.data[0][0] << "\n";

    return 0;
}