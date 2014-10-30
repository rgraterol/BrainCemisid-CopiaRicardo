#ifndef DIALOGSTATISTICS_H
#define DIALOGSTATISTICS_H

#include <QDialog>
#include "Class/statistic.h"

namespace Ui {
class DialogStatistics;
}

enum colums{EXECUTION,MAXTIME,MINTIME,AVERAGETIME};

class DialogStatistics : public QDialog
{
    Q_OBJECT
    
public:
    explicit DialogStatistics(QWidget *parent = 0);
    void setItems(Statistic *& statistic );
    inline QString averageTime(Statistic *&statistic, kernels kernel);
    inline QString minTime    (Statistic *&statistic, kernels kernel);
    inline QString maxTime    (Statistic *&statistic, kernels kernel);
    ~DialogStatistics();
    
private:
    Ui::DialogStatistics *ui;
};


#endif // DIALOGSTATISTICS_H
