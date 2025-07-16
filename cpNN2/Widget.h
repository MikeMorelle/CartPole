#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QTimer>
#include "Agent.h"

class Widget : public QWidget {
    Q_OBJECT

public:
    explicit Widget(Agent *sim, QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;

private slots:
    void updateSimulation();

private:
    Agent *sim;   // Zeiger auf gelerntes Sim-Objekt
    QTimer timer;
};

#endif
