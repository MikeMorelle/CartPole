#include "CartPoleSim.h"
#include <cmath>
#include <algorithm>

const double g = 9.81;
const double mass_cart = 1.0;
const double mass_pole = 0.1;
const double length = 0.5;

void CartPoleSim::step(double dt, int action) {
    double force = (action == 0) ? -10.0 : 10.0;

    double costheta = cos(theta);
    double sintheta = sin(theta);
    double total_mass = mass_cart + mass_pole;

    double temp = (force + mass_pole * length * omega * omega * sintheta) / total_mass;
    double theta_acc = (g * sintheta - costheta * temp) /
        (length * (4.0/3.0 - mass_pole * costheta * costheta / total_mass));
    double x_acc = temp - mass_pole * length * theta_acc * costheta / total_mass;

    x += v * dt;
    v += x_acc * dt;
    theta += omega * dt;
    omega += theta_acc * dt;
}

void CartPoleSim::reset() {
    x = 0;
    v = 0;
    theta = 0.05 * ((rng() % 2) * 2 - 1);
    omega = 0;
}

int CartPoleSim::getStateIndex() {
    int bins = 10;

    auto discretize = [bins](double val, double min, double max) {
        val = std::min(std::max(val, min), max);
        return static_cast<int>((val - min) / (max - min) * bins);
    };

    int dx = discretize(x, -2.4, 2.4);
    int dv = discretize(v, -2.0, 2.0);
    int dtheta = discretize(theta, -0.2, 0.2);
    int domega = discretize(omega, -2.0, 2.0);

    return dx + bins * (dv + bins * (dtheta + bins * domega));
}

int CartPoleSim::selectAction(double epsilon) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> actionDist(0, numActions - 1);

    if (dist(rng) < epsilon)
        return actionDist(rng);

    int state = getStateIndex();
    return (Q[state][0] > Q[state][1]) ? 0 : 1;
}

void CartPoleSim::updateQ(int state, int action, int reward, int nextState) {
    double alpha = 0.1;
    double gamma = 0.99;

    double q_predict = Q[state][action];
    double q_target = reward + gamma * std::max(Q[nextState][0], Q[nextState][1]);

    Q[state][action] += alpha * (q_target - q_predict);
}

bool CartPoleSim::isDone() const {
    return std::abs(x) > 2.4 || std::abs(theta) > 0.2;
}
