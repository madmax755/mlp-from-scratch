#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Sigmoid derivative
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// relu activation function
double relu(double x) {
    return std::max(x, 0.0);
}

// relu derivative
double relu_derivative(double x) {
    return (x>0) ? x : 0.0;
}


// read binary file into a vector
std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    
    if (file) {
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
        
        // for (unsigned char byte : buffer) {
        //     std::cout << static_cast<int>(byte) << " ";
        // }
        // std::cout << "\n";

        return buffer;
    }
    else {
        std::cout << "Error reading file " << path << "\n";
        
        return std::vector<unsigned char>();  // return an empty vector
    }
}


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

    // initialize matrix with zeros
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
        // weird loop order (k before j) makes more cache friendly
        for (size_t i = 0; i < rows; i++) {
            for (size_t k = 0; k < cols; k++) {
                for (size_t j = 0; j < other.cols; j++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // element-wise addition
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows or cols != other.cols) {
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
        if (rows != other.rows or cols != other.cols) {
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
        if (rows != other.rows or cols != other.cols) {
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

    // applies a function to every element in the array - for  passing in pointers
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    // applies a function to every element in the array - for passing in lambda functions and other callable objects
    template<typename Func>
    Matrix apply(Func func) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    Matrix softmax() const {
        Matrix result(rows, cols);

        for (size_t j = 0; j < cols; ++j) {
            double max_val = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < rows; ++i) {
                max_val = std::max(max_val, data[i][j]);
            }

            double sum = 0.0;
            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] = std::exp(data[i][j] - max_val);
                sum += result.data[i][j];
            }

            for (size_t i = 0; i < rows; ++i) {
                result.data[i][j] /= sum;
            }
        }

        return result;
    }
};

// layer class representing a single layer in the neural network
class Layer {
public:
    Matrix weights;
    Matrix bias;
    std::string activation_function;

    // constructor: initializes parameters
    Layer(size_t input_size, size_t output_size, std::string activation_function="sigmoid") 
        : weights(output_size, input_size),
          bias(output_size, 1),
          activation_function(activation_function) {
        weights.xavier_initialize();
    }

    // perform feedforward operation for this layer - returns the activations
    Matrix feedforward(const Matrix& input) {
        Matrix z = weights * input + bias;
        Matrix output(z.rows, z.cols);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        }
        else if (activation_function == "relu"){
            output = z.apply(relu);
        }
        else if (activation_function == "softmax") {
            output = z.softmax();
        }
        else {
            throw std::runtime_error("no activation function found for layer");
        }
        
        return output;
    }

    // perform feedforward operation for this layer - returns the activations AND PREACTIVATIONS for use in backpropagation
    std::vector<Matrix> feedforward_backprop(const Matrix& input) {
        Matrix z = weights * input + bias;
        Matrix output(z.rows, z.cols);
        if (activation_function == "sigmoid") {
            output = z.apply(sigmoid);
        }
        else if (activation_function == "relu"){
            output = z.apply(relu);
        }
        else if (activation_function == "softmax") {
            output = z.softmax();
        }
        else {
            throw std::runtime_error("no activation function found for layer");
        }
        
        return {output, z};
    }
};

// neural network class
class NeuralNetwork {
public:
    std::vector<Layer> layers;  // vector to store all layers
    struct EvaluationMetrics {
        double accuracy;
        double precision;
        double recall;
        double f1_score;
    };

    // constructor: create layers based on the given topology
    NeuralNetwork(const std::vector<int>& topology, const std::vector<std::string> activation_functions = {}) {
        if ((activation_functions.size() != topology.size()) and (activation_functions.size() != 0)) {
            throw std::invalid_argument("the size of activations_functions vector must be the same size as topology");
        }
        else if (activation_functions.size() == 0) {
            for (size_t i = 1; i < topology.size(); i++) {
                // do not pass in specific activation function - use the default specified in the layer constructor
                layers.emplace_back(topology[i-1], topology[i]);
            }
        }
        else {
            for (size_t i = 1; i < topology.size(); i++) {
                layers.emplace_back(topology[i-1], topology[i], activation_functions[i]);
            }
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

    // calculates the error gradients in the parameters for a single training example
    std::vector<std::vector<Matrix>> calculate_gradient(const Matrix& input, const Matrix& target) {
        // Forward pass
        std::vector<Matrix> activations = {input};
        std::vector<Matrix> preactivations = {input};

        for (auto& layer : layers) {
            auto results = layer.feedforward_backprop(activations.back());
            activations.push_back(results[0]);
            preactivations.push_back(results[1]);
        }

        // Backward pass
        int num_layers = layers.size();
        std::vector<Matrix> deltas;
        deltas.reserve(num_layers);

        // 1. Output layer error (δ^L = ∇_a C ⊙ σ'(z^L))

        Matrix output_delta = activations.back() - target;
        if (layers.back().activation_function == "sigmoid") {
            output_delta = output_delta.hadamard(preactivations.back().apply(sigmoid_derivative));
        }
        else if (layers.back().activation_function == "relu") {
            output_delta = output_delta.hadamard(preactivations.back().apply(relu_derivative));
        }
        else if (layers.back().activation_function == "none" or layers.back().activation_function == "softmax") {
            output_delta = output_delta;
        }
        else {
            throw std::runtime_error("Missing activation function during training");
        }
        

        deltas.push_back(output_delta);

        // 2. Hidden layer errors (δ^l = ((w^(l+1))^T δ^(l+1)) ⊙ σ'(z^l))
        for (int l = num_layers - 2; l >= 0; l--) {
            Matrix delta = (layers[l+1].weights.transpose() * deltas.back());

            if (layers[l].activation_function == "sigmoid") {
                delta = delta.hadamard(preactivations[l+1].apply(sigmoid_derivative)); // l+1 as preactivation contains the input while layers does not
            }
            else if (layers[l].activation_function == "relu") {
                delta = delta.hadamard(preactivations[l+1].apply(relu_derivative));
            }
            else if (layers[l].activation_function == "none") {
                delta = delta;
            }
            else {
                throw std::runtime_error("Missing activation function during training");
            }

            deltas.push_back(delta);
        }

        // Reverse deltas to match layer order
        std::reverse(deltas.begin(), deltas.end());

        // 3 & 4. calculate updates to weights and biases
        std::vector<std::vector<Matrix>> result;
        for (int l = 0; l < num_layers; l++) {
            // ∂C/∂b^l = δ^l
            // ∂C/∂w^l = δ^l (a^(l-1))^T
            Matrix weight_gradient = deltas[l] * activations[l].transpose();
            result.push_back({weight_gradient, deltas[l]});
        }
        
        return result;
    }

    // adjusts parameters using already computed error gradients
    void apply_adjustments(std::vector<std::vector<Matrix>>& gradients, double learning_rate) {
        int num_layers = layers.size();
        for (int l = 0; l < num_layers; l++) {
            layers[l].weights = layers[l].weights - (gradients[l][0] * learning_rate);
            layers[l].bias = layers[l].bias - (gradients[l][1] * learning_rate);
        }
    }

    // averages a vector of parameter error gradients
    std::vector<std::vector<Matrix>> average_gradients(const std::vector<std::vector<std::vector<Matrix>>>& gradients) {
        std::vector<std::vector<Matrix>> result;
        size_t no_layers = gradients[0].size();
        double no_examples = gradients.size();

        for (size_t i = 0; i<no_layers; ++i){
            auto layer = gradients[0][i];
            Matrix weight_average(layer[0].rows, layer[0].cols);
            Matrix bias_average(layer[1].rows, layer[1].cols);

            for (auto& example_gradients : gradients) {
                weight_average = weight_average + example_gradients[i][0];
                bias_average = bias_average + example_gradients[i][1];
            }

            weight_average = weight_average.apply([no_examples](double x) { return x / no_examples; });
            bias_average = bias_average.apply([no_examples](double x) { return x / no_examples; });

            result.push_back({weight_average, bias_average});
        }

        return result;
    }

    // trains the neural network (multithreaded)
    void train_mt(const std::vector<std::vector<Matrix>>& training_data, const std::vector<std::vector<Matrix>>& eval_data, int epochs, int batch_size, double learning_rate) {
        if (training_data.empty() or training_data[0].size() != 2) {
            throw std::invalid_argument("Training data must be a non-empty vector of vectors, each containing an input and a target matrix.");
        }     
        
        unsigned int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        std::vector<std::vector<std::vector<std::vector<Matrix>>>> thread_gradients(num_threads);
        std::mutex gradients_mutex;
        int counter = 0;

        // Create a vector of indices
        std::vector<size_t> indices(training_data.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Get a random seed
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);

        for (int epoch = 0; epoch < epochs; epoch++) {

            std::shuffle(indices.begin(), indices.end(), generator);

            for (size_t batch_start = 0; batch_start < training_data.size(); batch_start += batch_size) {
                
                // Clear previous gradients
                for (auto& grad : thread_gradients) {
                    grad.clear();
                }

                size_t current_batch_size = std::min(batch_size, static_cast<int>(training_data.size() - batch_start));

                // Spawn threads - splits the batch up between the number of threads
                for (unsigned int t = 0; t < num_threads; t++) {
                    threads[t] = std::thread([&, t, current_batch_size, batch_start]() {
                        size_t start = t * current_batch_size / num_threads;
                        size_t end = (t + 1) * current_batch_size / num_threads;
                        
                        std::vector<std::vector<std::vector<Matrix>>> local_gradients;
                        local_gradients.reserve(end-start);
                        for (size_t i = start; i < end; i++) {
                            size_t index = indices[batch_start + i];
                            if (index < training_data.size()) {
                                const auto& data_pair = training_data[index];
                                const Matrix& input = data_pair[0];
                                const Matrix& target = data_pair[1];

                                auto gradient = this->calculate_gradient(input, target);
                                local_gradients.push_back(gradient);
                            }
                        }                        
                        {
                            std::lock_guard<std::mutex> lock(gradients_mutex);
                            thread_gradients[t] = std::move(local_gradients);
                        }
                    });
                }


                // Join threads
                for (auto& thread : threads) {
                    thread.join();
                }


                // Flatten gradients
                std::vector<std::vector<std::vector<Matrix>>> batch_gradients;
                for (const auto& thread_grad : thread_gradients) {
                    batch_gradients.insert(batch_gradients.end(), thread_grad.begin(), thread_grad.end());
                }


                // Average gradients
                auto avg_gradient = average_gradients(batch_gradients);
                

                // Apply gradients
                this->apply_adjustments(avg_gradient, learning_rate);

                if (counter % 50 == 0) {
                    auto eval_results = evaluate_nn(eval_data);
                    std::cout << "----------------\naccuracy: " << eval_results.accuracy << "\n----------------\n";
                }
                counter++;
            }

            //todo ... print epoch results ...
        }
    }

    // gets the index of the maximum element in a nx1 matrix
    size_t get_index_of_max_element_in_nx1_matrix(const Matrix& matrix) {
        size_t index = 0;
        double max_value = matrix.data[0][0];
        for (size_t i = 1; i<matrix.rows; ++i) {
            if (matrix.data[i][0] > max_value) {
                index = i;
                max_value = matrix.data[i][0];
            }
        }
        return index;
    }

    // calculates the EvaluationMetrics on the inputted data
    EvaluationMetrics evaluate_nn(const std::vector<std::vector<Matrix>>& test_data) {
        if (test_data.empty() or test_data[0].size() != 2) {
            throw std::invalid_argument("Test data must be a non-empty vector of vectors, each containing an input and a target matrix.");
        }

        int true_positives = 0, false_positives = 0, false_negatives = 0;
        int total_correct = 0;
        size_t total_examples = test_data.size();

        for (const auto& example : test_data) {
            const Matrix& input = example[0];
            const Matrix& target = example[1];

            Matrix output = this->feedforward(input);

            // Assuming output and target are nx1 matrices
            size_t predicted_class = get_index_of_max_element_in_nx1_matrix(output);
            size_t actual_class = get_index_of_max_element_in_nx1_matrix(target);

            if (predicted_class == actual_class) {
                total_correct++;
                true_positives++;
            } else {
                false_positives++;
                false_negatives++;
            }
        }

        double accuracy = static_cast<double>(total_correct) / total_examples;
        
        // Avoid division by zero
        double precision = (true_positives + false_positives > 0) ? 
                           static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
        double recall = (true_positives + false_negatives > 0) ? 
                        static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;
        
        double f1_score = (precision + recall > 0) ? 
                          2 * (precision * recall) / (precision + recall) : 0.0;

        return {accuracy, precision, recall, f1_score};
    }
};

// Mean Squared Error (MSE) loss function
double mse_loss(const Matrix& predicted, const Matrix& target) {
    if (predicted.rows != target.rows or predicted.cols != target.cols) {
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

    std::vector<std::vector<Matrix>> training_set;
    training_set.reserve(9000);
    std::vector<std::vector<Matrix>> eval_set;
    training_set.reserve(1000);

    // create training set from binary image data files
    for (int i = 0; i < 10; ++i) {
        std::string file_path = "mnist data/data" + std::to_string(i) + ".dat";
        std::vector<unsigned char> full_digit_data = read_file(file_path);

        for (int j = 0; j < 784000; j += 28*28) { //todo make more general with training ratio
            std::vector<std::vector<double>> image_data;
                
            for (int k=0; k<28*28; ++k){
                double normalised_pixel = static_cast<double>(full_digit_data[j+k]) / 255.0;
                image_data.push_back({normalised_pixel});
            }

            // create the input matrix
            Matrix input_data(28*28, 1);
            input_data.data = image_data;

            // create the label matrix
            Matrix label_data(10,1);
            std::vector<std::vector<double>> data;

            // construct the label matrix with 1.0 in the postion of the digit and zeros elsewehre
            for (size_t l = 0; l<i; ++l) {data.push_back({0.0});}
            data.push_back({1.0});
            for (size_t l = 0; l+i+1<10; ++l) {data.push_back({0.0});}

            label_data.data = data;

            // push both image and label into training_set

            if (j<705600) {
                training_set.push_back({input_data, label_data});
            } else {
                eval_set.push_back({input_data, label_data});
            }
        }
    }

    // create the neural network
    int input_size = 28*28;
    std::vector<int> topology = {input_size, 32, 10};
    std::vector<std::string> activation_functions = {"none", "sigmoid", "softmax"};
    NeuralNetwork nn(topology, activation_functions);

    int batch_size = 128;
    int epochs = 10;
    double learning_rate = 0.2;

    // train the neural network
    nn.train_mt(training_set, eval_set, epochs, batch_size, learning_rate);

    return 0;
}