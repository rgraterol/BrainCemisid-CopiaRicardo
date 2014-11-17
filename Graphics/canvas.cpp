#include "canvas.h"

Canvas::Canvas(QWidget *parent):QGraphicsView(parent)
{
    item=new QGraphicsPixmapItem;
    item->setCacheMode(QGraphicsItem::DeviceCoordinateCache);

    scene= new QGraphicsScene(this);
    scene->addItem(item);
    setScene(scene);
    setDragMode(QGraphicsView::ScrollHandDrag);
}

void Canvas::loadImage()
{
    QPixmap image("./Obj/net.svg","svg");

    if(!image.isNull()){
        zoomReset();
        item->setPixmap(image);
    }
    else
        QMessageBox::information(this,tr("Error de entrada de datos"),
                                 tr("Imagen  no puede ser cargada"));
}

void Canvas::loadNumber(QImage *image)
{
    if(!image->isNull()){
        zoomReset();
        item->setPixmap(QPixmap::fromImage(*image));
    }
}


void Canvas::loadNumberCount(QImage image) {
    if(!image.isNull()){
        zoomReset();
        item->setPixmap(QPixmap::fromImage(image));
    }
}

void Canvas::zoomIn()
{
    scale(1.1,1.1);
}

void Canvas::zoomOut()
{
    scale(0.8,0.8);
}
void Canvas::zoomReset()
{
    resetTransform();
}
