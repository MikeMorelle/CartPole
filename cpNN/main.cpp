#include <QApplication>
#include <QMainWindow>
#include "Widget.h"
#include "Agent.h"
#include <fstream>
#include <vector>
#include <iostream>

int main(int argc, char *argv[]) {
    Agent sim;
    const int episodes = 1000;    // Episoden pro Bewertungsexperiment
    const int maxSteps = 200;	  //Zeitschritte
    const int numRuns = 1;        // Anzahl der Bewertungsexperimente (Wiederholungen)

    //Mittelwerte,Std und Schritte der einzelnen Runs speichern
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
    				reward = -20.0; 
			} else {
				// Belohnung, je besser Position und Winkel im Zielbereich
    				reward = 1.0
        			- 0.6 * std::abs(sim.getTheta()) / 0.7		// je kleiner der Winkel, desto besser (höher gewichtet)
        			- 0.4 * std::abs(sim.getX()) / 2.4;		// je näher an der Mitte, desto besser
    				reward = std::max(0.0, reward);

    				if (inTargetZone)
        				reward += 1.0; // Bonus on top
			}

             		// Neuronales Netz mit Q-Learning aktualisieren
			sim.updateQ(stateVec, action, reward, nextStateVec);

             		totalReward += reward;
             		++steps;
			
                	// Update Zustand für nächsten Schritt
                	stateVec = nextStateVec;
        	}

        	// Exploration verringern – der Agent nutzt mehr das Gelernte
       		epsilon = std::max(min_epsilon, epsilon * decay_rate);
		// Schritte dieser Episode speichern
        	stepsPerEpisode.push_back(steps);
		std::cout << "Run " << run << " | Episode " << ep << " | Schritte: " << steps << "\n";
    	}


    }

    // Nach dem Training als CSV speichern
    std::ofstream out("learning_curve.csv");
    for (size_t i = 0; i < stepsPerEpisode.size(); ++i) {
        out << i << "," << stepsPerEpisode[i] << "\n";
    }
    out.close();

    // GUI starten
    QApplication app(argc, argv);
    QMainWindow window;
    Widget *widget = new Widget();
    window.setCentralWidget(widget);
    window.resize(600, 400);
    window.setWindowTitle("CartPole Simulator mit Q-Learning");
    window.show();

    return app.exec();
}
