#include "dialogtablebns.h"
#include "ui_dialogtablebns.h"

DialogTableBNS::DialogTableBNS(QWidget *parent, const NeuralNetwork *neuralSenses, const SizeNet *sizeNet) :
    QDialog(parent),
    ui(new Ui::DialogTableBNS)
{
    ui->setupUi(this);
    this->neuralSenses = neuralSenses;

    fillLineEdits(sizeNet);
    fillAllTables();
}

void DialogTableBNS::fillLineEdits( const SizeNet *sizeNet)
{
    int mem = sizeNet->sizevectorFlags + sizeNet->sizeVectorNeuron + sizeNet->sizeBinaryCharacteristic;
    int maxNeuron= sizeNet->numNeuron;

    ui->lineEditNumNeuronSight->setText(QString::number(maxNeuron));
    ui->lineEditMemSight->setText(QString::number(mem));
    ui->lineEditPtrSight->setText(QString::number(*(neuralSenses[SIGHT].ptr)));

    ui->lineEditMemHearing->setText(QString::number(mem));
    ui->lineEditNumNeuronHearing->setText(QString::number(maxNeuron));
    ui->lineEditPtrHearing->setText(QString::number(*(neuralSenses[HEARING].ptr)));
}

void DialogTableBNS::fillAllTables()
{
    fillTable( &neuralSenses[ SIGHT   ] , ui->tableWidgetSight);
    fillTable( &neuralSenses[ HEARING ] , ui->tableWidgetHearing);
    connect(ui->tableWidgetSight,SIGNAL(doubleClicked(QModelIndex))  ,this,SLOT(reciveRowSight(QModelIndex)));
    connect(ui->tableWidgetHearing,SIGNAL(doubleClicked(QModelIndex)),this,SLOT(reciveRowHearing(QModelIndex)));
}

void DialogTableBNS::fillTable(const NeuralNetwork *neuralSenses, QTableWidget *&table)
{
    int numNeuron = *(neuralSenses->ptr);

    if(numNeuron == 0){
        table->hide();
        return;
    }

    createHeaderRow(table,numNeuron);

    for (int id = 0; id < numNeuron; id++)
    {
        QString vector;
        QTableWidgetItem * newItem = new QTableWidgetItem(QString::number(id));
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        table->setItem(id, ID, newItem);

        newItem = new QTableWidgetItem(QString::number(neuralSenses->vectorFlags[id*9+CAT]));
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        table->setItem(id,CATEGORY, newItem);

        newItem = new QTableWidgetItem(QString::number(neuralSenses->vectorFlags[id*9+RAT]));
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        table->setItem(id,RATIO, newItem);

        for (int j = 0; j < 32; j++) {

            vector.append(QString::number(neuralSenses->vectorNeuron[id * 32+j]));
            if( j < 31 )
                vector.append(",");
        }

        newItem = new QTableWidgetItem(vector);
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        table->setItem(id,CENTER,newItem);

    }

    table->resizeColumnToContents(ID);
    table->resizeColumnToContents(CATEGORY);
    table->resizeColumnToContents(CENTER);
}

void DialogTableBNS::createHeaderRow(QTableWidget *table, int rows)
{
    QStringList list;
    table->setRowCount(rows);

    for (int i = 0; i < rows; i++)
        list << QString::number(i);

    table->setVerticalHeaderLabels(list);
}

DialogTableBNS::~DialogTableBNS()
{
    delete ui;
}

void DialogTableBNS::reciveRowSight(QModelIndex index)
{
    if(index.column()==CENTER)
        paintBinaryCharacteristic(SIGHT,ui->tableWidgetSight->item(index.row(),ID)->text().toInt());    
}

void DialogTableBNS::reciveRowHearing(QModelIndex index)
{
    if(index.column()==CENTER)
        paintBinaryCharacteristic(HEARING, ui->tableWidgetHearing->item(index.row(),ID)->text().toInt());    
}

void DialogTableBNS::paintBinaryCharacteristic(senses sense, int ptr)
{
    unsigned short  displacement = 8 * sizeof (unsigned short) -1;
    unsigned short  mask=  1 << displacement;


    QImage * image = new QImage(QSize(400,400), QImage::Format_MonoLSB);
    image->fill(Qt::color1);

    QPainter paint;
    paint.begin(image);

    QPen pen(QColor(Qt::color1));
    paint.setPen(pen);
    paint.setBrush(QBrush(QColor(Qt::color1), Qt::SolidPattern));
    QString text= (sense == HEARING ) ? "VOZ" : "PENSANDO";
    paint.drawText(QRect(100,20,100,100),text);

    for (int i = 0; i < 16 ; i++) {

        unsigned short  value = neuralSenses[sense].binaryCharacteristic[ ptr * 16 +i ];

        for (unsigned short j = 0 ; j <= displacement; j++) {

            if(value &mask)
                paint.drawRect(QRect(100+(15-j)*10,50+i*10,10,10));//(x,y) x columnas y filas

            value <<= 1;
        }
    }

    paint.end();

    ViewFinder &view = ViewFinder::getInstance();
    view.showBinaryCharacteristic(image);

}

