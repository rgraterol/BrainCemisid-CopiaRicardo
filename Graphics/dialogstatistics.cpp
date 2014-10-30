#include "dialogstatistics.h"
#include "ui_dialogstatistics.h"

DialogStatistics::DialogStatistics(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogStatistics)
{
    ui->setupUi(this);
}

void DialogStatistics::setItems( Statistic *& statistic)
{
    ui->tableWidget->item(BOOT,EXECUTION)->setText(QString::number(statistic[BOOT].numExecutions));
    ui->tableWidget->item(BOOT,MAXTIME)->setText(maxTime(statistic,BOOT));
    ui->tableWidget->item(BOOT,MINTIME)->setText(minTime(statistic,BOOT));
    ui->tableWidget->item(BOOT,AVERAGETIME)->setText(averageTime(statistic,BOOT));

    ui->tableWidget->item(RECOGNIZE,EXECUTION)->setText(QString::number(statistic[RECOGNIZE].numExecutions));
    ui->tableWidget->item(RECOGNIZE,MAXTIME)->setText(maxTime(statistic,RECOGNIZE));
    ui->tableWidget->item(RECOGNIZE,MINTIME)->setText(minTime(statistic,RECOGNIZE));
    ui->tableWidget->item(RECOGNIZE,AVERAGETIME)->setText(averageTime(statistic,RECOGNIZE));

    ui->tableWidget->item(CORRECT,EXECUTION)->setText(QString::number(statistic[CORRECT].numExecutions));
    ui->tableWidget->item(CORRECT,MAXTIME)->setText(maxTime(statistic,CORRECT));
    ui->tableWidget->item(CORRECT,MINTIME)->setText(minTime(statistic,CORRECT));
    ui->tableWidget->item(CORRECT,AVERAGETIME)->setText(averageTime(statistic,CORRECT));

    ui->tableWidget->item(RESET,EXECUTION)->setText(QString::number(statistic[RESET].numExecutions));
    ui->tableWidget->item(RESET,MAXTIME)->setText(maxTime(statistic,RESET));
    ui->tableWidget->item(RESET,MINTIME)->setText(minTime(statistic,RESET));
    ui->tableWidget->item(RESET,AVERAGETIME)->setText(averageTime(statistic,RESET));
}

QString DialogStatistics::averageTime(Statistic *& statistic ,kernels kernel)
{
    return (statistic[kernel].numExecutions == 0 ) ? "---" :
                                                   QString::number(statistic[kernel].accumulateTime/statistic[kernel].numExecutions);
}

QString DialogStatistics::minTime(Statistic *&statistic, kernels kernel)
{
       return (statistic[kernel].numExecutions == 0 ) ? "---" :
                                                   QString::number(statistic[kernel].minTime);
}

QString DialogStatistics::maxTime(Statistic *&statistic, kernels kernel)
{
    return (statistic[kernel].numExecutions == 0 ) ? "---" :
                                                   QString::number(statistic[kernel].maxTime);
}

DialogStatistics::~DialogStatistics()
{
    delete ui;
}
