#include "viewfinder.h"

ViewFinder* ViewFinder::singleton=NULL;

ViewFinder::ViewFinder( QMainWindow * mainWindow )
{    
    dialogCanvas=new QMainWindow(mainWindow);
    canvas=NULL;

    zoomIn = new QAction(QIcon("icons/zoomin.png"),QMainWindow::tr("A&cercar"),dialogCanvas);
    zoomIn->setShortcut(QKeySequence::ZoomIn);
    zoomIn->setStatusTip(QMainWindow::tr("Acercar imagen"));

    zoomOut = new QAction(QIcon("icons/zoomout.png"),QMainWindow::tr("&Alejar"),dialogCanvas);
    zoomOut->setShortcut(QKeySequence::ZoomOut);
    zoomOut->setStatusTip(QMainWindow::tr("Alejar imagen"));

    zoomReset = new QAction(QIcon("icons/zoomreset.png"),QMainWindow::tr("&Normal"),dialogCanvas);
    zoomReset ->setShortcut(QMainWindow::tr("Ctrl+0"));
    zoomReset ->setStatusTip(QMainWindow::tr("Estado Original de imagen"));

    toolBar = new QToolBar("Barra de Herramientas");
    toolBar ->setMovable(false);
    toolBar ->setIconSize(QSize(32,32));

    toolBar->addAction(zoomIn);
    toolBar->addAction(zoomOut);
    toolBar->addAction(zoomReset);

    dialogCanvas->addToolBar(toolBar);

    int width = 600, height = 540;
    dialogCanvas->setMinimumSize(width,height);
    dialogCanvas->setWindowTitle("Visor Grafico");
    dialogCanvas->setWindowIcon(QIcon("icons/seeNet"));
}


ViewFinder::~ViewFinder()
{
    freeAll();
}

ViewFinder &ViewFinder::getInstance(QMainWindow *main)
{
    if(singleton==NULL)
        singleton= new ViewFinder(main);

    return *singleton;
}

void ViewFinder::showNet()
{
    createCanvas();
    QTimer::singleShot(200, canvas, SLOT(loadImage()));
    QTimer::singleShot(250, dialogCanvas, SLOT(show()));
}

void ViewFinder::showBinaryCharacteristic(QImage *image)
{
    createCanvas();
    canvas->loadNumber(image);
    QTimer::singleShot(250, dialogCanvas, SLOT(show()));
}

void ViewFinder::createCanvas()
{
    if(canvas!=NULL)
       delete canvas;

    canvas= new Canvas();

    dialogCanvas->setCentralWidget(canvas);
    QObject::connect(zoomIn, SIGNAL(triggered()),canvas, SLOT(zoomIn()));
    QObject::connect(zoomOut, SIGNAL(triggered()),canvas, SLOT(zoomOut()));
    QObject::connect(zoomReset , SIGNAL(triggered()),canvas, SLOT(zoomReset()));
}

void ViewFinder::freeAll()
{
    if(canvas!=NULL)
        delete canvas;

    delete(dialogCanvas);
}
