#ifndef DIALOGTABLEBNS_H
#define DIALOGTABLEBNS_H

#include <QDialog>
#include <QTableWidget>
#include "Class/neuralNetwork.h"
#include "viewfinder.h"

namespace Ui {
class DialogTableBNS;
}

enum {ID,CATEGORY,RATIO,CENTER};
class DialogTableBNS : public QDialog
{
    Q_OBJECT
public slots:
    void reciveRowSight  (QModelIndex index);
    void reciveRowHearing(QModelIndex index);

public:
    explicit DialogTableBNS (QWidget *parent , const NeuralNetwork * neuralSenses,const SizeNet * sizeNet );
    void     fillLineEdits  (const SizeNet * sizeNet);
    void     fillAllTables  ();
    void     fillTable      (const NeuralNetwork *neuralSenses,QTableWidget *&table );
    void     createHeaderRow(QTableWidget * table,int rows);
    void     paintBinaryCharacteristic(senses sense, int ptr);
    ~DialogTableBNS();
    
private:
    Ui::DialogTableBNS *ui;
    const NeuralNetwork * neuralSenses;
    QMainWindow * ptrMain;
};

#endif // DIALOGTABLEBNS_H
