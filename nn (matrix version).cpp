#include <vector>
#include <iostream>
#include <cmath>
#include <random>
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
    void randomize() {
        std::random_device rd;  // obtain a random number from hardware
        std::mt19937 gen(rd());  // seed the generator
        std::uniform_real_distribution<> dis(-1.0, 1.0);  // define the range
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = dis(gen);  // generate random number
            }
        }
    }

    // matrix multiplication overloading the * operator 
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
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

    // apply a function element-wise to the matrix
    void apply(double (*func)(double)) {
        for (auto& row : data) {
            for (auto& elem : row) {
                elem = func(elem);
            }
        }
    }
};

// sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// derivative of sigmoid function (used in backpropagation)
double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

// layer class representing a single layer in the neural network
class Layer {
public:
    Matrix weights;
    Matrix bias;
    Matrix output;
    // storing 'vectors' as matrix types to reuse matmul code (mathematicians close your eyes).

    // constructor: initialize weights and biases with random values
    Layer(size_t input_size, size_t output_size) : weights(output_size, input_size), bias(output_size, 1), output(output_size, 1) {
        weights.randomize();
        bias.randomize();
    }

    // perform feedforward operation for this layer
    Matrix feedforward(const Matrix& input) {
        output = weights * input + bias; 
        output.apply(sigmoid);
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
    NeuralNetwork nn({2, 3, 1});
    
    Matrix input(2, 1);
    input.data = {{1}, {0}};  // example input
    Matrix output = nn.feedforward(input);
    
    std::cout << "Output: " << output.data[0][0] << "\n";
    return 0;
}