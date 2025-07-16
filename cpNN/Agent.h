#ifndef AGENT_H
#define AGENT_H

#include <vector>
#include <random>
#include "SimpleNN.h"

class Agent {
public:
    double x = 0, v = 0, theta = 0.05, omega = 0;

    void step(double dt, int action);
    void reset();
    std::vector<double> getNormalizedState();
    int selectAction(double epsilon);
    void updateQ(const std::vector<double>& stateVec, int action, double reward, const std::vector<double>& nextStateVec);

    bool isDone() const;

    Agent();

    double getX() const;
    double getTheta() const;

private:
    const int bins = 10;
    const int numStates = bins * bins * bins * bins;
    const int numActions = 2;
    std::vector<std::vector<double>> Q;
    std::default_random_engine rng;
    SimpleNN qnet{4,24,2,0.001}; // z.B. 4 Eingänge (x, v, θ, ω), 4 Hidden, 2 Outputs (Q-Werte), Lernrate
};

#endif

