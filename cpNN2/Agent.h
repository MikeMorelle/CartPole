#pragma once
#include "SimpleNN.h"
#include <vector>
#include <random>
#include <deque>
#include <tuple>

class Agent {
public:
    Agent();

    void step(double dt, int action);
    void reset();
    int selectAction(double epsilon);
    void updateQ(const std::vector<double>& stateVec, int action, double reward, const std::vector<double>& nextStateVec, bool done);
    bool isDone() const;

    double getX() const;
    double getTheta() const;
    std::vector<double> getNormalizedState();

    // Target-Network-Aktualisierung
    void updateTargetNetwork();

    // Variablen für Target-Netzwerk und Replay
    SimpleNN qnet;
    SimpleNN target_qnet;

    std::deque<std::tuple<std::vector<double>, int, double, std::vector<double>, bool>> memory;
    std::mt19937 rng;

    size_t memory_capacity = 10000;
    int train_start = 1000;
    int batch_size = 64;
    int update_target_interval = 10;
	
    double x, v, theta, omega;
    const int numActions = 2;
    const int inputSize = 4;

private:

};