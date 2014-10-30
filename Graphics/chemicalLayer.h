#ifndef CHEMICALLAYER_H
#define CHEMICALLAYER_H

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QMouseEvent>
#include <QGraphicsRectItem>
#include <QColorDialog>
#include <QSettings>
#include <QApplication>
#include <math.h>


using namespace std;
class ChemicalLayer : public QGraphicsView {
    Q_OBJECT

public :
    ChemicalLayer(int amountWidth =16,int amountHeight= 16,int width = 10,int height = 10);
    ~ChemicalLayer();

    void intializeInterface(int amountWidth,int amountHeight,int width,int height);
    unsigned char * generateCharacteristic();
    unsigned short * generateBinaryCharacteristic();
    void paintPattern(const int * vector, const int &numRows);
    void crateLayer();

    QList<QList<bool> >& state();
    void setState(int x,int y,bool s);
    int width();
    int height();
    QColor activeColor();
    QColor inactiveColor();
    void newActiveColor();
    void newInactiveColor();
    void setDefaultColor();
    void setActiveLayer(bool activeLayer = true);
    bool getNoData();

    bool getIsEditable() const;
    void setIsEditable(bool value);

signals:
    void change();

public slots :
    void clear();

protected :
    void mouseMoveEvent(QMouseEvent* event);
    void mousePressEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);

private :
    QGraphicsRectItem* getNode(const QPoint& pos);
    bool* getNodeState(const QPoint& pos);
    QGraphicsScene* m_scene;
    QList<QList<QGraphicsRectItem*> > m_grid;
    QList<QList<bool> > m_state;

    bool m_leftMousePressed;
    bool m_rightMousePressed;
    bool isEditable;
    bool activeLayer;
    bool noData;

    int m_amountWidth;
    int m_amountHeight;
    int m_width;
    int m_height;
    int countRectActivate;

    QColor m_activeColor;
    QColor m_inactiveColor;
};

#endif
