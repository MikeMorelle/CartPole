#include <QApplication>
#include <QMainWindow>
#include "CartPoleWidget.h"
#include "CartPoleSim.h"
#include <fstream>
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QMainWindow window;

    CartPoleSim sim;
    const int episodes = 1000;    // Episoden pro Bewertungsexperiment
    const int maxSteps = 200;	  //Zeitschritte
    const int numRuns = 20;        // Anzahl der Bewertungsexperimente (Wiederholungen)

    //Mittelwerte,Std und Schritte der einzelnen Runs speichern
    std::vector<double> means(numRuns);
    std::vector<double> stddevs(numRuns);
    std::vector<int> stepsPerEpisode;

    //ε-greedy Parameter
    double epsilon = 1.0;
    const double min_epsilon = 0.1;
    const double decay_rate = 0.999;


    for (int run = 0; run < numRuns; ++run) {
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
                		reward = -10.0;
             		}else {
               			 //Belohnungen relativ zur Winkelabweichung & Positionsabweichung mit Gewichtungen
                		reward = 1.0 - (std::abs(sim.getTheta()) / 0.7) * 0.7 - (std::abs(sim.getX()) / 2.4) * 0.3;
				reward = std::max(0.0, reward);  // niemals negativ

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

        	// ε mit decay verringern (weniger Exploration über Zeit)
       		epsilon = std::max(min_epsilon, epsilon * decay_rate);
        	stepsPerEpisode.push_back(steps);
    	}

    	// Mittelwert berechnen
    	double meanSteps = std::accumulate(stepsPerEpisode.begin(), stepsPerEpisode.end(), 0.0) / stepsPerEpisode.size();

    	// Standardabweichung berechnen
    	double sumSquaredDiffs = 0.0;
    	for (int s : stepsPerEpisode) {
        	sumSquaredDiffs += (s - meanSteps) * (s - meanSteps);
    	}
    	double stddevSteps = std::sqrt(sumSquaredDiffs / stepsPerEpisode.size());

    	means[run] = meanSteps;
    	stddevs[run] = stddevSteps;

   	 std::cout << "\n[Run " << run << "] Ergebnis:\n"
              << "Durchschnittliche Balancierzeit: " << meanSteps << " Schritte\n"
              << "Standardabweichung: " << stddevSteps << " Schritte\n";

    }

    // Nach dem Training: Lernkurve als CSV speichern
    std::ofstream out("learning_curve.csv");
    for (size_t i = 0; i < stepsPerEpisode.size(); ++i) {
        out << i << "," << stepsPerEpisode[i] << "\n";
    }
    out.close();

    // GUI starten
    CartPoleWidget *widget = new CartPoleWidget(&sim);
    window.setCentralWidget(widget);
    window.resize(600, 400);
    window.setWindowTitle("CartPole Simulator mit Q-Learning");
    window.show();

    return app.exec();
}
