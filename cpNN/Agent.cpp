#include "Agent.h"
#include "SimpleNN.h"
#include <cmath>
#include <algorithm>

// physikalische Konstanten
const double g = 9.81;			//Gravitation [m/s^2]
const double mass_cart = 1.0;		//Masse des Wagens [kg]
const double mass_pole = 0.1;		//Masse des Stabs [kg]
const double length = 0.5;		//Länge des Stabs [m]

Agent::Agent() {
    // Initialisiere Q-Tabelle (bei NN optional)
    Q.resize(numStates, std::vector<double>(numActions, 0.0));
}

//getter für differenziertere Belohnungen
double Agent::getX() const {
    return x;
}

double Agent::getTheta() const {
    return theta;
}

// Ausführen eines Simulationsschrittes
void Agent::step(double dt, int action)  {
    // ausübende Kraft: 0 = links (-10N), 1 = rechnts (+10N)
    double force = (action == 0) ? -10.0 : 10.0;

    // Gesamtmasse, Kosinus und Sinus des aktuellen Stabwinkels
    double costheta = cos(theta);
    double sintheta = sin(theta);
    double total_mass = mass_cart + mass_pole;

    // Zwischenwert für Berechnungen: horizontale Kraft auf den Wagen (aus externer Kraft + Stabrotation)
    double temp = (force + length * omega * omega * sintheta) / total_mass;

    // Winkelbeschleunigung resultiert aus Drehmomentbilanz am Stab (4/3 als Kompensationsfaktor, der aus Trägheitsmoment des Stabs stammt (angenommen als homogener Stab, am unteren Ende verankert)
    double theta_acc = (g * sintheta - costheta * temp) /
        (length * (4.0/3.0 - mass_pole * costheta * costheta / total_mass));

    // Wagenbeschleunigung
    double x_acc = temp - length * theta_acc * costheta / total_mass;

    // Zustand aktualisieren
    x += v * dt;		// Wagenposition
    v += x_acc * dt;            // Wagengeschwindigkeit
    theta += omega * dt;	// Stabwinkelposition
    omega += theta_acc * dt;    // Winkelgeschwindigkeit
}

// reset nach jeder Episode
void Agent::reset() {
    //zufälliger Startpunkt zwischen -50 un +50 cm
    std::uniform_real_distribution<double> x_dist(-0.5, 0.5);
    x = x_dist(rng);
    v = 0;                     // ohne Anfangswagengeschwindigkeit

    // zufällige leichte Neigung zwischen -0.5 und +0.5 rad ≈ ±28°
    std::uniform_real_distribution<double> t_dist(-0.5, 0.5);
    theta = t_dist(rng);

    omega = 0;                  //ohne Anfangsstabgeschwindigkeit
}

//normierter Zustand für Eingabe ins NN
std::vector<double> Agent::getNormalizedState() {
    return {
        x / 2.4,        
        v / 3.0,
        theta / 0.7,
        omega / 5.0
    };
}

//ε-greedy (Zufall oder beste bekannte Entscheidung)
int Agent::selectAction(double epsilon) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> actionDist(0, numActions - 1);

    //Zufallszahl < ε: zufällige Aktion (Exploration)
    if (dist(rng) < epsilon)
        return actionDist(rng);

    //ansonsten: beste bekannte Aktion wählen → Exploitation
    std::vector<double> q_values = qnet.predict(getNormalizedState());
    return std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()));
}

//Q-Lean via Backprop in NN
void Agent::updateQ(const std::vector<double>& stateVec, int action, double reward, const std::vector<double>& nextStateVec) {
    //schätze zukünftige Belohnung (Q-Wert) im nächsten Zustand
    std::vector<double> q_next = qnet.predict(nextStateVec);
    double max_q_next = *std::max_element(q_next.begin(), q_next.end());

    //Zielwert für das aktuelle Paar (state, action)
    double target = reward + 0.99 * max_q_next;			//discount-Faktor y =0.99

    //trainiere das neuronale Netz so, dass es Q(state, action) ≈ target lernt
    qnet.train(stateVec, action, target); 
}

//fertig?
bool Agent::isDone() const {
    return std::abs(x) > 2.4             //Wagen verlässt erlaubte Zone
       || std::abs(theta) > 0.7;         //Stab kippt zu weit
}
