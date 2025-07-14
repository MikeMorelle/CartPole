#include <QApplication>
#include <QMainWindow>
#include "CartPoleWidget.h"
#include "CartPoleSim.h"
#include <fstream>
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
    CartPoleSim sim;
    const int episodes = 10000;

    std::vector<int> stepsPerEpisode;

    for (int ep = 0; ep < episodes; ++ep) {
        sim.reset();
        int state = sim.getStateIndex();
        int steps = 0;

        while (!sim.isDone() && steps < 500) {
            int action = sim.selectAction(0.1); // epsilon-greedy
            sim.step(0.01, action);
            int reward = sim.isDone() ? -1 : 1;
            int nextState = sim.getStateIndex();
            sim.updateQ(state, action, reward, nextState);
            state = nextState;
            ++steps;
        }

        stepsPerEpisode.push_back(steps);
        std::cout << "Episode " << ep << ": " << steps << " steps\n";
    }

    // Nach dem Training: Lernkurve als CSV speichern
    std::ofstream out("learning_curve.csv");
    for (size_t i = 0; i < stepsPerEpisode.size(); ++i) {
        out << i << "," << stepsPerEpisode[i] << "\n";
    }
    out.close();

    // GUI starten
    QApplication app(argc, argv);
    QMainWindow window;
    CartPoleWidget *widget = new CartPoleWidget();
    window.setCentralWidget(widget);
    window.resize(600, 400);
    window.setWindowTitle("CartPole Simulator mit Q-Learning");
    window.show();

    return app.exec();
}
