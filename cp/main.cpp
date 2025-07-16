#include <QApplication>
#include <QMainWindow>
#include "CartPoleWidget.h"
#include "CartPoleSim.h"
#include <fstream>
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
    CartPoleSim sim;
    const int episodes = 1000;		//Läufe pro Bewertungsexperiment
    std::vector<int> stepsPerEpisode;

    //ε-greedy Parameter
    double epsilon = 1.0;
    const double min_epsilon = 0.01;
    const double decay_rate = 0.995;

    for (int ep = 0; ep < episodes; ++ep) {
        sim.reset();
        int state = sim.getStateIndex();
        int steps = 0;
        double totalReward = 0.0;

        while (!sim.isDone() && steps < 200) {
             int action = sim.selectAction(epsilon); // Aktion wählen mit epsilon-greedy
             sim.step(0.01, action);

             double reward = 0.0;
	     bool inTargetZone = std::abs(sim.getX()) < 0.05 && std::abs(sim.getTheta()) < 0.05;
	     bool failed = sim.isDone();

	     //bestrafe, wenn nicht im Zielbereich
             if (failed) {
    		reward = -1.0;
             } else {
                //Belohnungen relativ zur Winkelabweichung & Positionsabweichung mit Gewichtungen
    		reward = 1.0 - (std::abs(sim.getTheta()) / 0.7) * 0.6  - (std::abs(sim.getX()) / 2.4) * 0.4;

    		if (inTargetZone) {
        	    reward += 1.0;  // Bonus on top
    		}
	     }

             //Q(s,a) aktualisieren
             int nextState = sim.getStateIndex();
             sim.updateQ(state, action, reward, nextState);
             state = nextState;

             totalReward += reward;
             ++steps;
        }

        // ε mit decy verringern (weniger Exploration über Zeit)
        epsilon = std::max(min_epsilon, epsilon * decay_rate);
        stepsPerEpisode.push_back(steps);

	//log
        std::cout << "Episode " << ep
                  << " | Steps: " << steps
                  << " | Epsilon: " << epsilon
                  << " | Reward: " << totalReward
                  << std::endl;
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
