#include "Agent.h"
#include "SimpleNN.h"
#include <cmath>
#include <algorithm>

// physikalische Konstanten
const double g = 9.81;			//Gravitation [m/s^2]
const double mass_cart = 1.0;		//Masse des Wagens [kg]
const double mass_pole = 0.1;		//Masse des Stabs [kg]
const double length = 0.5;		//Länge des Stabs [m]

Agent::Agent()
    : qnet(4, 6, 2, 0.001), target_qnet(4, 6, 2, 0.001) {
    rng.seed(std::random_device{}());
    updateTargetNetwork();
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
        omega / 1.0
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
void Agent::updateQ(const std::vector<double>& stateVec, int action, double reward, const std::vector<double>& nextStateVec, bool done) {
    if (memory.size() >= memory_capacity) memory.pop_front();
    memory.emplace_back(stateVec, action, reward, nextStateVec, done);

    if (memory.size() < train_start) return;

    std::vector<std::tuple<std::vector<double>, int, double, std::vector<double>, bool>> batch;
    std::sample(memory.begin(), memory.end(), std::back_inserter(batch), batch_size, rng);

// Sammle Inputs und Ziel-Q-Werte für Mini-Batch
std::vector<std::vector<double>> state_batch;
std::vector<std::vector<double>> qvalue_batch;

for (auto& [s, a, r, s_next, done] : batch) {
    std::vector<double> q_values = qnet.predict(s);
    std::vector<double> q_next = target_qnet.predict(s_next);
    double max_q_next = *std::max_element(q_next.begin(), q_next.end());

    double target = done ? r : r + 0.99 * max_q_next;
    q_values[a] = target;

    state_batch.push_back(s);
    qvalue_batch.push_back(q_values);
}

// Jetzt trainiere alle gleichzeitig
for (size_t i = 0; i < state_batch.size(); ++i) {
    for (int action = 0; action < qvalue_batch[i].size(); ++action) {
        qnet.train(state_batch[i], action, qvalue_batch[i][action]);
    }
}
}

void Agent::updateTargetNetwork() {
    target_qnet.copyFrom(qnet);
}

//fertig?
bool Agent::isDone() const {
    return std::abs(x) > 2.4             //Wagen verlässt erlaubte Zone
       || std::abs(theta) > 0.7;         //Stab kippt zu weit
}