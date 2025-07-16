#include <QApplication>
#include <QMainWindow>
#include "Widget.h"
#include "Agent.h"
#include <fstream>
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QMainWindow window;

    Agent sim;
    const int episodes = 1000;    // Episoden pro Bewertungsexperiment
    const int maxSteps = 200;	  //Zeitschritte
    const int numRuns = 20;        // Anzahl der Bewertungsexperimente (Wiederholungen)

    //Mittelwerte,Std und Schritte der einzelnen Runs speichern
    std::vector<double> means(numRuns);
    std::vector<double> stddevs(numRuns);
    std::vector<int> stepsPerEpisode;

    //ε-greedy Parameter: Exploration vs. Exploitation
    double epsilon = 1.0;		//ANfang: viel Zufall
    const double min_epsilon = 0.1;
    const double decay_rate = 0.999;


    for (int run = 0; run < numRuns; ++run) {
	epsilon = 1.0;
	stepsPerEpisode.clear();			// Neue Liste für jeden Run

	for (int ep = 0; ep < episodes; ++ep) {
                sim.reset();
		std::vector<double> stateVec = sim.getNormalizedState();	//Zustand vor dem Schritt
        	int steps = 0;
        	double totalReward = 0.0;

        	while (!sim.isDone() && steps < 200) {
             		int action = sim.selectAction(epsilon); 		// Aktion wählen (Zufall oder gelernt)
             		sim.step(0.01, action);					// Simulation für einen Schritt = 10 ms

			std::vector<double> nextStateVec = sim.getNormalizedState();	// Nächster Zustand

             		double reward = 0.0;
             		bool inTargetZone = std::abs(sim.getX()) < 0.05 && std::abs(sim.getTheta()) < 0.05;
             		bool failed = sim.isDone();

             		//bestrafe, wenn umfällt
             		if (failed) {
    				reward = -1.0; 
			} else {
    				if (inTargetZone)
        				reward = 10.0; // Bonus
				else {

					reward = 2.0
    						- 0.6 * std::abs(sim.getTheta()) / 0.7
   						- 0.4 * std::abs(sim.getX()) / 2.4;
					reward = std::max(0.0, reward);

				}

			}

             		// Neuronales Netz mit Q-Learning aktualisieren
			sim.remember(stateVec, action, reward, nextStateVec, failed);
 			sim.trainFromReplay();

             		totalReward += reward;
             		++steps;
			
                	// Update Zustand für nächsten Schritt
                	stateVec = nextStateVec;
        	}

        	// Exploration verringern – der Agent nutzt mehr das Gelernte
       		epsilon = std::max(min_epsilon, epsilon * decay_rate);
		// Schritte dieser Episode speichern
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

    // Nach dem Training als CSV speichern
    std::ofstream out("learning_curve.csv");
    for (size_t i = 0; i < stepsPerEpisode.size(); ++i) {
        out << i << "," << stepsPerEpisode[i] << "\n";
    }
    out.close();

    // GUI starten
    Widget *widget = new Widget(&sim);
    window.setCentralWidget(widget);
    window.resize(600, 400);
    window.setWindowTitle("CartPole Simulator mit Q-Learning");
    window.show();

    return app.exec();
}