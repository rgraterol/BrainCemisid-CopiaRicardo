#ifndef VIEWFINDER_H
#define VIEWFINDER_H
#include <QMainWindow>
#include <QAction>
#include <QToolBar>
#include <iostream>
#include "canvas.h"

class ViewFinder
{

private:

    ViewFinder(QMainWindow *mainWindow);
    static ViewFinder * singleton;
    QMainWindow       * dialogCanvas;
    Canvas   * canvas;
    QAction  * zoomIn;
    QAction  * zoomOut;
    QAction  * zoomReset;
    QToolBar * toolBar;

public:

    static ViewFinder & getInstance(QMainWindow *main = 0);
    void showNet();
    void showBinaryCharacteristic(QImage *image);
    void think(QImage image);
    void createCanvas();
    void freeAll();
    ~ViewFinder();

};

#endif // VIEWFINDER_H
