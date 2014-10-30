#include "formteaching.h"
#include "ui_formteaching.h"

FormTeaching::FormTeaching(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FormTeaching)
{
    ui->setupUi(this);
    init();
    createSignals();
    QRegExp alphabet("(\\d|=|\\+)");
    ui->lineEditInput->setValidator(new QRegExpValidator(alphabet,this));
}

void FormTeaching::setState(stateNeuralNetwork state, int category)
{
    switch (state) {

    case IS_HIT:
        isHit(category);
        break;

    case NO_HIT:
        noHit();
        break;

    case DIFF:
        isDiff();
        break;

    default:
        init();
        break;
    }
}

QLineEdit *FormTeaching::getLineEditInput()
{
    return ui->lineEditInput;
}

QLineEdit *FormTeaching::getLineEditOut()
{
    return ui->lineEditOut;
}

QPushButton *FormTeaching::getPtrButtonTeach()
{
    return ui->pushButtonTeach;
}

QPushButton *FormTeaching::getPtrButtonState()
{
    return ui->pushButtonState;
}

QPushButton *FormTeaching::getPtrButtonGoodAnswer()
{
    return ui->pushButtonGoodAnswer;
}

FormTeaching::~FormTeaching()
{
    delete ui;
}

void FormTeaching::changeInput()
{
    if(ui->lineEditInput->text().isEmpty())
        ui->pushButtonTeach->setEnabled(false);

    else
        ui->pushButtonTeach->setEnabled(true);
}

void FormTeaching::goodAnswer()
{
    init();
    ui->labelEmoticon->setPixmap(QPixmap("icons/happy.png"));
    QTimer::singleShot(2000, this, SLOT(finishTime()));
}

void FormTeaching::finishTime()
{
    ui->labelEmoticon->setPixmap(QPixmap("icons/sleep.png"));
    emit signalGoodAnswer(this->senseTeaching);
}

void FormTeaching::badAnswer()
{
    init();
    ui->labelQuestion->setVisible(true);
    ui->lineEditInput->setVisible(true);
    ui->pushButtonTeach->setVisible(true);
    ui->pushButtonTeach->setEnabled(false);
    ui->labelEmoticon->setPixmap(QPixmap("icons/question.png"));
}

void FormTeaching::init()
{

    this->setFixedHeight(155);
    ui->labelQuestion->setVisible(false);
    ui->labelSense->setVisible(false);
    ui->labelCorrectTest->setVisible(false);
    ui->lineEditInput->setVisible(false);
    ui->lineEditOut->setVisible(false);
    ui->lineEditInput->setText("");
    ui->lineEditOut->setText("");
    ui->pushButtonState->setVisible(false);
    ui->pushButtonBadAnswer->setVisible(false);
    ui->pushButtonGoodAnswer->setVisible(false);
    ui->lineEditOut->setAlignment(Qt::AlignCenter);
    ui->lineEditInput->setAlignment(Qt::AlignCenter);
    ui->pushButtonTeach->setVisible(false);
    ui->lineEditOut->setReadOnly(true);
    ui->labelEmoticon->setPixmap(QPixmap("icons/sleep.png"));
}

void FormTeaching::createSignals()
{
    connect(ui->lineEditInput       ,SIGNAL(textEdited(QString)),this,SLOT(changeInput()));
    connect(ui->pushButtonBadAnswer ,SIGNAL(clicked())          ,this,SLOT(badAnswer ()));
    connect(ui->pushButtonGoodAnswer,SIGNAL(clicked())          ,this,SLOT(goodAnswer()));
}

void FormTeaching::isHit(int category)
{   
    QString text(caracterCla(category));
    ui->labelQuestion->setVisible(false);
    ui->lineEditInput->setVisible(false);
    ui->labelSense->setVisible(true);
    ui->labelCorrectTest->setVisible(true);
    ui->lineEditOut->setVisible(true);
    ui->lineEditOut->setText(text);
    ui->pushButtonState->setVisible(true);
    ui->pushButtonBadAnswer->setVisible(true);
    ui->pushButtonGoodAnswer->setVisible(true);
    ui->labelEmoticon->setPixmap(QPixmap("icons/response.svg"));
}

void FormTeaching::noHit()
{
    ui->labelQuestion->setVisible(true);
    ui->labelSense->setVisible(true);
    ui->lineEditInput->setVisible(true);
    ui->lineEditOut->setVisible(true);
    ui->lineEditOut->setText("Ignoro");
    ui->pushButtonTeach->setVisible(true);
    ui->pushButtonTeach->setEnabled(false);
    ui->labelEmoticon->setPixmap(QPixmap("icons/question.png"));
}

void FormTeaching::isDiff()
{
    ui->pushButtonState->setVisible(true);
    ui->labelQuestion->setVisible(true);
    ui->labelSense->setVisible(true);
    ui->lineEditInput->setVisible(true);
    ui->lineEditOut->setVisible(true);
    ui->lineEditOut->setText("Confundido!");
    ui->pushButtonTeach->setVisible(true);
    ui->pushButtonTeach->setEnabled(false);
    ui->labelEmoticon->setPixmap(QPixmap("icons/question.png"));
}

char FormTeaching::caracterCla(int category)
{
    switch (category) {

    case 0:
        return('0');

    case 1:
        return('1');

    case 2:
        return('2');

    case 3:
        return('3');

    case 4:
        return('4');

    case 5:
        return('5');

    case 6:
        return('6');

    case 7:
        return('7');

    case 8:
        return('8');

    case 9:
        return('9');

    case'=':
        return('=');

    case'+':
        return('+');

    default:
        return('e');
    }
}

senses FormTeaching::getSenseTeaching() const
{
    return senseTeaching;
}

void FormTeaching::setSenseTeaching(const senses &value)
{
    senseTeaching = value;
}
