#include "CartPoleSim.h"
#include <cmath>
#include <algorithm>

// physikalische Konstanten
const double g = 9.81;			//Gravitation [m/s^2]
const double mass_cart = 1.0;		//Masse des Wagens [kg]
const double mass_pole = 0.1;		//Masse des Stabs [kg]
const double length = 0.5;		//Länge des Stabs [m]

CartPoleSim::CartPoleSim() {
    Q.resize(numStates, std::vector<double>(numActions, 0.0));
}

//getter für differenziertere Belohnungen
double CartPoleSim::getX() const {
    return x;
}

double CartPoleSim::getTheta() const {
    return theta;
}

// Ausführen eines Simulationsschrittes
void CartPoleSim::step(double dt, int action)  {
    // ausübende Kraft: 0 = links (-10N), 1 = rechnts (+10N)
    double force = (action == 0) ? -10.0 : 10.0;

    // Gesamtmasse, Kosinus und Sinus des aktuellen Stabwinkels
    double costheta = cos(theta);
    double sintheta = sin(theta);
    double total_mass = mass_cart + mass_pole;

    // Zwischenwert für Berechnungen: horizontale Kraft auf den Wagen (aus externer Kraft + Zentrifugalkraft des Stabs)
    double temp = (force + length * omega * omega * sintheta) / total_mass;

    // Winkelbeschleunigung resultiert aus Drehmomentbilanz am Stab (4/3 als Kompensationsfaktor, der aus Trägheitsmoment des Stabs stammt (angenommen als homogener Stab, am unteren Ende verankert)
    double theta_acc = (g * sintheta - costheta * temp) /
        (length * (4.0/3.0 - mass_pole * costheta * costheta / total_mass));

    // Wagenbeschleunigung resultiert aus Newton II
    double x_acc = temp - length * theta_acc * costheta / total_mass;

    // Zustand aktualisieren
    x += v * dt;		// Wagenposition
    v += x_acc * dt;            // Wagengeschwindigkeit
    theta += omega * dt;	// Stabwinkelposition
    omega += theta_acc * dt;    // Winkelgeschwindigkeit
}

// reset nach jeder Episode
void CartPoleSim::reset() {
    //zufälliger Startpunkt zwischen -50 un +50 cm
    std::uniform_real_distribution<double> x_dist(-0.5, 0.5);
    x = x_dist(rng);
    v = 0;                     // ohne Anfangswagengeschwindigkeit
    // zufällige leichte Neigung zwischen -0.5 und +0.5 rad ≈ ±28°
    std::uniform_real_distribution<double> t_dist(-0.5, 0.5);
    theta = t_dist(rng);

    omega = 0;                  //ohne Anfangsstabgeschwindigkeit
}

// Diskretisierung des kontinuierlichen Zustands in Q-Tabellenindexe
int CartPoleSim::getStateIndex() {
    int bins = this->bins;      //Anzahl der STufen pro Zustand

    //wandelt kontinuierliche Wert in int
    auto discretize = [bins](double val, double min, double max) {
        val = std::min(std::max(val, min), max);                                  //Wert begrenzen zwischen ±Komponente
	int result = static_cast<int>((val - min) / (max - min) * bins);          //Skalieren auf bins und Indexvergabe (v-m: bringt Wert in pos. Bereich; /max-min: zw. 0-1; *bins skaliert auf 0-bins)
        return std::min(result, bins -1);                                         //Sicherstellen, dass nicht über Q-Table hinaus, sonst Segmentation Fault
    };

    //Diskretisierung jeder Komponente
    int dx = discretize(x, -2.4, 2.4);
    int dv = discretize(v, -3.0, 3.0);
    int dtheta = discretize(theta, -0.7, 0.7);
    int domega = discretize(omega, -5.0, 5.0);

    //kodiert 4D-Zustand als 1-D Index für Q-Table
    return dx + bins * (dv + bins * (dtheta + bins * domega));
}

//ε-greedy (Zufall oder beste bekannte Entscheidung)
int CartPoleSim::selectAction(double epsilon) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> actionDist(0, numActions - 1);

    //Zufallszahl < ε: zufällige Aktion (Exploration)
    if (dist(rng) < epsilon)
        return actionDist(rng);

    //ansonsten: beste Aktion (Exploitation)
    int state = getStateIndex();
    return (Q[state][0] > Q[state][1]) ? 0 : 1;
}

//Q-Leanring update: lerne aus aktueller Erfahrung
void CartPoleSim::updateQ(int state, int action, int reward, int nextState) {
    double alpha = 0.1;                                                         //Lernrate (wie stark wird gelernt)
    double gamma = 0.99;                                                        //Diskonierungsfaktor für zukünftige Belohnungen

    double q_predict = Q[state][action];                                        //aktuelle Q-Schätzung
    double q_target = reward + gamma * std::max(Q[nextState][0], Q[nextState][1]); //erwartete Belohnung

    //Updateregel: Q_neu = Q_alt + α * (Ziel - Schätzung)
    Q[state][action] += alpha * (q_target - q_predict);
}

//fertig?
bool CartPoleSim::isDone() const {
    return std::abs(x) > 2.4             //Wagen verlässt erlaubte Zone
       || std::abs(theta) > 0.7;         //Stab kippt zu weit
}