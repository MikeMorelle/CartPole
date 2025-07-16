#include "Widget.h"
#include <QPainter>
#include <cmath>

Widget::Widget(Agent *sim, QWidget *parent)
    : QWidget(parent), sim(sim) {
    this->sim->reset();
    connect(&timer, &QTimer::timeout, this, &Widget::updateSimulation);
    timer.start(16);		//etwa 60FPS
}

void Widget::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    int w = width();
    int h = height();

    painter.drawLine(0, h / 2 + 50, w, h / 2 + 50);

    int cartX = static_cast<int>(w / 2 + sim->x * 100);
    int cartY = h / 2;

    QRect cartRect(cartX - 25, cartY, 50, 30);
    painter.setBrush(Qt::gray);
    painter.drawRect(cartRect);

    double angle = sim->theta;
    int poleLength = 100;
    int endX = cartX + static_cast<int>(poleLength * sin(angle));
    int endY = cartY - static_cast<int>(poleLength * cos(angle));
    painter.setPen(QPen(Qt::red, 4));
    painter.drawLine(cartX, cartY, endX, endY);
}

void Widget::updateSimulation() {
    if (sim->isDone()) {
	sim->reset();
    }

    int action = sim->selectAction(0.0);
    sim->step(0.01, action);
    update();
}
