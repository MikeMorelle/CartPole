#ifndef AGENT_H
#define AGENT_H

#include <vector>
#include <random>
#include "SimpleNN.h"
#include <deque>

class Agent {

struct Transition {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> nextState;
    bool done;
};

public:
    double x = 0, v = 0, theta = 0.05, omega = 0;

    void step(double dt, int action);
    void reset();
    std::vector<double> getNormalizedState();
    int selectAction(double epsilon);
    bool isDone() const;

    Agent();

    double getX() const;
    double getTheta() const;

    void remember(const std::vector<double>& state, int action, double reward, const std::vector<double>& nextState, bool done);
    void trainFromReplay();

private:
    const int numActions = 2;
    std::default_random_engine rng;
    SimpleNN qnet; // z.B. 4 Eingänge (x, v, θ, ω), 12 Hidden, 2 Outputs (Q-Werte), Lernrate

    std::deque<Transition> replayBuffer;
    const size_t bufferSize = 10000;
    const int batchSize = 32;
};

#endif