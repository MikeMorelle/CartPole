#ifndef CARTPOLESIM_H
#define CARTPOLESIM_H

#include <vector>
#include <random>

class CartPoleSim {
public:
    double x = 0, v = 0, theta = 0.05, omega = 0;

    void step(double dt, int action);
    void reset();
    int getStateIndex();
    int selectAction(double epsilon);
    void updateQ(int state, int action, int reward, int nextState);

    bool isDone() const;

private:
    const int numStates = 10 * 10 * 10 * 10;
    const int numActions = 2;
    std::vector<std::vector<double>> Q = std::vector<std::vector<double>>(numStates, std::vector<double>(numActions, 0.0));
    std::default_random_engine rng;
};

#endif
