#ifndef SIMPLENN_H
#define SIMPLENN_H

#include <vector>
#include <random>

class SimpleNN {
public:
    SimpleNN(int input_size_, int hidden_size_, int output_size_, double lr = 0.01);
    std::vector<double> predict(const std::vector<double>& input);
    void train(const std::vector<double>& input, int action, double target);

private:
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;

    std::vector<std::vector<double>> W1, W2;
    std::vector<double> b1, b2;
    std::vector<double> hidden_layer, output_layer;
	std::vector<double> hidden_layer_raw;

    std::mt19937 rng;

    double relu(double x);
    double relu_derivative(double x);
};

#endif