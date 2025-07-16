#include <QApplication>
#include <QMainWindow>
#include "Widget.h"
#include "Agent.h"
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QMainWindow window;

    Agent sim;
    const int episodes = 1000;
    const int maxSteps = 200;
    const int numRuns = 1;

    std::vector<int> stepsPerEpisode;
    double epsilon = 1.0;
    const double min_epsilon = 0.1;
    const double decay_rate = 0.995;

    for (int run = 0; run < numRuns; ++run) {
        epsilon = 1.0;
        stepsPerEpisode.clear();

        for (int ep = 0; ep < episodes; ++ep) {
            sim.reset();
            auto state = sim.getNormalizedState();
            int steps = 0;

            while (!sim.isDone() && steps < maxSteps) {
                int action = sim.selectAction(epsilon);
                sim.step(0.01, action);
                auto nextState = sim.getNormalizedState();

                bool inTarget = std::abs(sim.getX()) < 0.05 && std::abs(sim.getTheta()) < 0.05;
                bool done = sim.isDone();
double reward = 0.0;
if (done) {
    reward = -10.0;
} else {
reward = 1.0 - std::pow(sim.getTheta(), 2) - 0.25 * std::pow(sim.getX(), 2);
reward = std::clamp(reward, -1.0, 1.0);
}

                sim.updateQ(state, action, reward, nextState, done);
                state = nextState;
                steps++;
            }

            epsilon = std::max(min_epsilon, epsilon * decay_rate);
            stepsPerEpisode.push_back(steps);

            if (ep % 10 == 0) {
    sim.updateTargetNetwork();
}
            std::cout << "Run " << run << " | Episode " << ep << " | Schritte: " << steps << "\n";
        }
    }

    std::ofstream out("learning_curve.csv");
    for (size_t i = 0; i < stepsPerEpisode.size(); ++i)
        out << i << "," << stepsPerEpisode[i] << "\n";
    out.close();

    Widget *widget = new Widget(&sim);
    window.setCentralWidget(widget);
    window.resize(600, 400);
    window.setWindowTitle("CartPole Simulator mit DQN");
    window.show();

    return app.exec();
}