#include "SimpleNN.h"
#include <cmath>
#include <algorithm>
#include <iostream>

SimpleNN::SimpleNN(int input_size_, int hidden_size_, int output_size_, double lr)
    : input_size(input_size_), hidden_size(hidden_size_), output_size(output_size_), learning_rate(lr)
{
    rng.seed(std::random_device{}());
    std::uniform_real_distribution<double> dist(-0.1, 0.1);	//Gewichte zwischen -1 und 1

    // Eingabe → Hidden Layer Gewichte W1 zufällig
    W1.resize(hidden_size, std::vector<double>(input_size));
    for (auto& row : W1)
        for (auto& w : row) w = dist(rng);
   
    //biases für Hidden
    b1.resize(hidden_size);
    for (auto& b : b1) b = dist(rng);

    // Hidden → Output Layer Gewichte W2 zufällig
    W2.resize(output_size, std::vector<double>(hidden_size));
    for (auto& row : W2)
        for (auto& w : row) w = dist(rng);

    // Biases für Output Layer
    b2.resize(output_size);
    for (auto& b : b2) b = dist(rng);
}

//Aktivierungsfunktion = nur positive Werte
double SimpleNN::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

//Ableitung für Backpropagation
double SimpleNN::relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

//vorwärts
std::vector<double> SimpleNN::predict(const std::vector<double>& input) {

    //1. Input → Hidden
    hidden_layer.resize(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        double sum = b1[i];				//starte mit bias
        for (int j = 0; j < input_size; ++j)
            sum += W1[i][j] * input[j];			//gewichtete Summe y = x*w
        hidden_layer[i] = relu(sum);			//f(y)
    }

    //2. Hidden → Output
    output_layer.resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        double sum = b2[i];
    for (int j = 0; j < hidden_size; ++j) 
        sum += W2[i][j] * hidden_layer[j];
 
        output_layer[i] = sum;				//linear, keine Aktfkt. für Regression/Q-Werte
    }

    return output_layer;				//Output = Q-Werte für jede Aktion
}


//trainiere mit Backprop
void SimpleNN::train(const std::vector<double>& input, int action, double target) {
    predict(input);						//vorwärts -> speichert hidden und output

    //Fehler = gewünschter Zielwert - tatsächlicher Q-Wert
    double error = target - output_layer[action];

    //Fehlervektor für gewählte Aktion
    std::vector<double> d_output(output_size, 0.0);
    d_output[action] = error;

    //zurück von Output → Hidden Layer (SGB)
    std::vector<double> d_hidden(hidden_size, 0.0);

    //1. w und b im Output Layer anpassen
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            double grad = d_output[i] * hidden_layer[j];	//gradient = Fehler x Aktivierung
            W2[i][j] -= learning_rate * grad;			//w update
        }
        b2[i] -= learning_rate * d_output[i];			//b update
    }

    //2. Fehler an Hidden Layer zurück
    for (int j = 0; j < hidden_size; ++j) {
        double grad_sum = 0.0;
        for (int i = 0; i < output_size; ++i) {
            grad_sum += d_output[i] * W2[i][j];			//Fehlerfluss zurückrechnen
        }
        d_hidden[j] = grad_sum * relu_derivative(hidden_layer[j]);  //ReLu
    }

    //3. w und b im Hidden Layer anpassen
    for (int j = 0; j < hidden_size; ++j) {
        for (int k = 0; k < input_size; ++k) {
            double grad = d_hidden[j] * input[k];	//Fehler x Eingabewert
            W1[j][k] -= learning_rate * grad;
      }
        b1[j] -= learning_rate * d_hidden[j];		//bias anpassen
    }
}