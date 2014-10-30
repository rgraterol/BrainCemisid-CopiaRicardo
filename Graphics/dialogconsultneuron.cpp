#include "dialogconsultneuron.h"
#include "ui_dialogconsultneuron.h"

DialogConsultNeuron::DialogConsultNeuron(const unsigned char *ptrHearing,const unsigned char* ptrSight, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogConsultNeuron)
{
    ui->setupUi(this);
    ptrNeuron = new unsigned char [2];
    ptrNeuron[HEARING] = *ptrHearing;
    ptrNeuron[ SIGHT ] = *ptrSight;
    ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);

    if(ptrNeuron[ SIGHT ] == 0)
    {
        ui->radioButtonSight->setEnabled(false);
        ui->radioButtonHearing->setChecked(true);
        ui->radioButtonHearing->clicked(true);
    }

    else if(ptrNeuron[ HEARING ] == 0)
    {
        ui->radioButtonHearing->setEnabled(false);
        ui->radioButtonSight->clicked(true);
    }

    else
        ui->radioButtonSight->clicked(true);
}

senses DialogConsultNeuron::radioButtonActive()
{
    if( ui->radioButtonSight->isChecked())
        return SIGHT;

    return HEARING;
}

int DialogConsultNeuron::returnIdNeuron()
{
    return(ui->lineEdit->text().toInt());
}

void DialogConsultNeuron::setLimits(QString text)
{
    ui->labelLimits->setText(text);
}

void DialogConsultNeuron::setRegularExp(unsigned char number)
{
    QString exp("\\d{0,");
    QString stringNumber= QString::number( number );
    exp.append(QString::number( stringNumber.size() ));
    exp.append("}");
    QRegExp regExp(exp);
    ui->lineEdit->setValidator(new QRegExpValidator(regExp,this));
}

DialogConsultNeuron::~DialogConsultNeuron()
{
    delete ui;
}

QString DialogConsultNeuron::textLimits(const unsigned char number)
{
    QString text("<font color ='red'> Introduzca un numero natural menor a ");
    text.append(QString::number(number));
    text.append(" </ font>");
    return text;
}


void DialogConsultNeuron::changeSight()
{
    ui->lineEdit->clear();
    ui->labelLimits->setText(textLimits(ptrNeuron[SIGHT]));
    setRegularExp(ptrNeuron[ SIGHT ]);
}

void DialogConsultNeuron::changeHearing()
{
    ui->lineEdit->clear();
    ui->labelLimits->setText(textLimits(ptrNeuron[HEARING]));
    setRegularExp(ptrNeuron[ HEARING ]);
}

void DialogConsultNeuron::checkText()
{
    if(ui->lineEdit->text().isEmpty()){
        ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        return;
    }

    int maxLimit;
    int value = ui->lineEdit->text().toInt();

    if(ui->radioButtonSight->isChecked())
        maxLimit=ptrNeuron[SIGHT];

    else
        maxLimit=ptrNeuron[HEARING];

    if(value < 0 || value >= maxLimit)
        ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);

    else
        ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
}
