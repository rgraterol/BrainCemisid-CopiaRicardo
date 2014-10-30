#ifndef FORMTEACHING_H
#define FORMTEACHING_H

#include <QWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QTimer>
#include "Class/neuralNetwork.h"

namespace Ui {
class FormTeaching;
}

class FormTeaching : public QWidget
{
    Q_OBJECT
    
public:
    explicit FormTeaching(QWidget *parent = 0);
    void setState(stateNeuralNetwork state,int category = -1);
    QPushButton * getPtrButtonTeach();
    QPushButton * getPtrButtonState();
    QPushButton * getPtrButtonGoodAnswer();
    QLineEdit * getLineEditInput();
    QLineEdit * getLineEditOut();
    void createSignals();
    ~FormTeaching();

    senses getSenseTeaching() const;
    void setSenseTeaching(const senses &value);

signals:
    void signalGoodAnswer(senses sense);

public slots:
    void init();
    void changeInput();
    void goodAnswer();
    void finishTime();
    void badAnswer();

private:
    Ui::FormTeaching *ui;
    senses senseTeaching;
    void isHit(int category);
    void noHit();
    void isDiff();
    char caracterCla(int category);
};

#endif // FORMTEACHING_H
