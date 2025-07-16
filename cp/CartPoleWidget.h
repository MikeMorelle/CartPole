#ifndef CARTPOLEWIDGET_H
#define CARTPOLEWIDGET_H

#include <QWidget>
#include <QTimer>
#include "CartPoleSim.h"

class CartPoleWidget : public QWidget {
    Q_OBJECT

public:
    explicit CartPoleWidget(CartPoleSim *sim, QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;

private slots:
    void updateSimulation();

private:
    CartPoleSim *sim;   // Zeiger auf gelerntes Sim-Objekt
    QTimer timer;
};

#endif
