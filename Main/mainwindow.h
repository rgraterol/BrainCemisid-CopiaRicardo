#ifndef MAINWINDOW_H
#define MAINWINDOW_H
// QT includes áéíóú
#include <QMainWindow>
#include <QProcess>
#include <QMessageBox>
#include <QString>

//c++
#include <iostream>
#include <fstream>
#include <new>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDA/helper_cuda.h>

// My includes
#include "Graphics/dialogconsultneuron.h"
#include "Graphics/dialogselecthardware.h"
#include "Graphics/dialogstatistics.h"
#include "Graphics/dialogtablebns.h"
#include "Graphics/formteaching.h"
#include "Graphics/chemicalLayer.h"
#include "Graphics/viewfinder.h"
#include "Class/neuralNetwork.h"
#include "Class/interface.h"
#include "Class/statistic.h"
#include "Class/relationNetwork.h"

#include "Class/countNetwork.h"
#include "Class/culturalNet.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
signals:
    void cross();

public slots :
    void showSelectDevice(bool isVisibleButton = true);
    void processGrid();
    void clearTables();
    void learnHearing();
    void learnSight();
    void resetHearing();
    void resetSight();
    void activateButtonProcess();
    void activeLayers(bool active = false );
    void finishGoodAnswer(senses sense);
    void paintNetNeuron(senses sense,bool onlyHits=true);
    void paintBinaryCharacteristic(senses sense,int ptr);
    void paintStateHearing();
    void paintStateSight();
    void paintHearing();
    void paintSight();
    void activateMainWindow(bool activate = true);
    void showDialogConsult();
    void showDialogStatistic();
    void showDialogTableNeuron();
    void aboutBrainCemisid();
    void aboutCuda();
    void launchWave();
    void runCrossing();

    ///////Trabajo de Ricardo//////////
    void countProtocol();
    void orderProtocol();
    void activateButtonBip();
    void startCount();
    void stopCount();


private slots:
    void on_pushButtonBip_clicked();
    void on_checkBox_cuento_clicked();

private:
    Ui::MainWindow *     ui;
    QImage         *     image;
    ChemicalLayer  *     chemicalLayerEar;
    ChemicalLayer  *     chemicalLayerEye;
    FormTeaching   **    formsTeaching;
    DialogConsultNeuron  * dialogConsult;
    DialogSelectHardware * dialogSelectHardware;
    DialogStatistics     * dialogStatistics;
    DialogTableBNS       * dialogTable;

    NeuralNetwork      * neuralSenses;
    Statistic          * statistics;
    stateNeuralNetwork * stateSenses;
    Interface          * interface;
    SizeNet              sizeNet;

    /////TRABAJO RICARDO/////
    CountNetwork * countNetwork;
    OrderNetwork * orderNetwork;


    cudaDeviceProp       deviceProp;

    RelationNetwork *relationSenses;
    CulturalNet *addNet;

    bool               * isInactivateSense;
    int numSenses;
    unsigned char    selectedDevice;
    unsigned char  * characteristicVectorEar;
    unsigned char  * characteristicVectorEye;
    unsigned short * characteristicBinaryVector;

    void initGui();
    void activateInterface(bool state);
    void setNull();
    void freeVectorsCharacteristic();
    void freeFormTeaching();
    void freeStates();
    void freeSenses();
    void freeInterface();
    void freeUi();
    void freeMem();
    void generateVectorsCharacteristic();
    void createTablesCharacteristic();
    void initializeTable();
    void intitializeSenses(int numSenses = 2); // Dos sentidos vista y oido
    void createStringForTable();
    void earTraining();
    void senseTraining(QString nameFileCategories, QString nameFileWaves, QString path, int numPatterns, senses sense);
    void setDataTable(QStringList listStringRow, QStringList listStringColumn);
    int  calculateNumNeuron();
    void createInterfacesTeaching();
    void setFormsCheck(bool state = false);
    void learn(senses sense);
    void realLearn(senses sense);
    void learnBinaryCharacteristic(senses sense, int ptr);
    void generateDot(QString nameFile, senses sense,bool onlyHits = true);
    void generatePng(QString nameFile);
    void showWarning(QString windowTitle,QString message);
    unsigned char returnCategory(QString cad);
    unsigned char * returnVectorCategory(senses sense);
    bool  multipleDevice();

    //TESIS JULIO
    void actualiceCategory(unsigned char, unsigned char);
    bool ambiguity(unsigned char );
    bool analyticsNeuron ();
    void buildRelation (unsigned char);
    void initializeRelation(int);
    unsigned char checkInRelationNet();
    void think(senses senses);
    void initializeCuturalNet(int);
    void freeCulturalNet();



    //templates
    template < class T >
    void freeGenericPtr(T * ptr);

    //funciones para depurar
    void printAllVectorSenses();
    void printVectorSenses(senses sense);
    void printIdsNeuronHit(senses sense);
    void printSense(senses sense);
    void printCategory  (senses sense);


    ////Trabajo Ricardo///
    void printCountNetwork();
    void paintCount(senses sense, int ptr, int times);
    char caracterCla(int category);
};

#endif // MAINWINDOW_H
