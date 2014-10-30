#ifndef CANVAS_H
#define CANVAS_H

#include <QtGui>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QMessageBox>

class Canvas: public QGraphicsView
{
    Q_OBJECT

    QGraphicsScene *scene;
    QGraphicsPixmapItem *item;

public:

    Canvas(QWidget *parent = 0);

public slots:
    void loadImage();
    void loadNumber(QImage * image);
    void zoomIn();
    void zoomOut ();
    void zoomReset ();
};

#endif // CANVAS_H
