#include "dialogselecthardware.h"
#include "ui_dialogselecthardware.h"

DialogSelectHardware::DialogSelectHardware(QWidget *parent, int selectDevice) :
    QDialog(parent),
    ui(new Ui::DialogSelectHardware)
{
    deviceCount        = 0;
    deviceProp         = NULL;
    this->selectDevice = selectDevice;

    ui->setupUi(this);
    connect(ui->deviceComboBox , SIGNAL(currentIndexChanged(int)) , this , SLOT(ChangeText(int)));
    connect(ui->pushButton     , SIGNAL(clicked())                , this , SLOT(setDevice()));
    setListDevice();
    ChangeText(0);
}

void DialogSelectHardware::setVisibleSelectButton(bool active)
{
    if(active)
        setWindowFlags(Qt::WindowMaximizeButtonHint);

    ui->pushButton->setVisible(active);
}

void DialogSelectHardware::showWarning(QString windowTitle, QString message)
{
    QMessageBox::warning(0,windowTitle,message,QMessageBox::Ok);
}

void DialogSelectHardware::addItem(QString text, int row, int col)
{
    QTableWidgetItem * newItem = new QTableWidgetItem(text);
    newItem->setTextAlignment(Qt::AlignVCenter);
    newItem->setTextAlignment(Qt::AlignLeft);
    ui->tableWidget->setItem(row,col, newItem);
}

void DialogSelectHardware::setListDevice()
{
    cudaGetDeviceCount(&deviceCount);

    QString text("Detectados "+QString::number(deviceCount)+" Dispositivos Compatibles con CUDA");
    QMessageBox::information(0,"Dispositivos Detectados",text,QMessageBox::Ok);

    deviceProp = new cudaDeviceProp;

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaGetDeviceProperties(deviceProp, dev);

        QString text("Device "+QString::number(dev).append(" : ")+ deviceProp->name);
        ui->deviceComboBox->addItem(text);
    }
}

DialogSelectHardware::~DialogSelectHardware()
{
    delete ui;

    if(deviceProp != NULL)
        delete deviceProp;
}

int DialogSelectHardware::getIndexSelectecDevice()
{
    return selectDevice;
}

void DialogSelectHardware::ChangeText(int indexDevice)
{
    int  driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(indexDevice);
    cudaGetDeviceProperties(deviceProp, indexDevice);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    char msg[256];
    SPRINTF(msg,"%.0f MBytes (%llu bytes)\n",
            (float)deviceProp->totalGlobalMem/1048576.0f, (unsigned long long) deviceProp->totalGlobalMem);

    ui->tableWidget->clear();
    addItem(QString ("Device "+QString::number(indexDevice).append(" : ")+ deviceProp->name),0,0);
    addItem((selectDevice == indexDevice) ? "Dispositivo Seleccionado " : " ",0,1);
    addItem("CUDA Driver Version / Runtime Version",1,0);
    addItem(QString ("%1.%2  /  %3.%4").arg(driverVersion/1000).arg((driverVersion%100)/10).arg( runtimeVersion/1000).arg((runtimeVersion%100)/10),1,1);
    addItem("CUDA Capability Major/Minor version number: ",2,0);
    addItem(QString ("%1.%2").arg(deviceProp->major).arg(deviceProp->minor),2,1);
    addItem("Total amount of global memory:",3,0);
    addItem(msg,3,1);
    addItem(QString ("(%1) Multiprocessors, (%2) CUDA Cores/MP:%3 CUDA Cores").arg( deviceProp->multiProcessorCount).arg( _ConvertSMVer2Cores(deviceProp->major, deviceProp->minor)).arg( _ConvertSMVer2Cores(deviceProp->major, deviceProp->minor) * deviceProp->multiProcessorCount),4,0);
    addItem("Total amount of constant memory:",5,0);
    addItem(QString ("%1 bytes").arg(deviceProp->totalConstMem),5,1);
    addItem("Total amount of shared memory per block:",6,0);
    addItem(QString ("%1 bytes").arg(deviceProp->sharedMemPerBlock),6,1);
    addItem("Total number of registers available per block:",7,0);
    addItem(QString ("%1").arg(deviceProp->regsPerBlock),7,1);
    addItem("Warp size:",8,0);
    addItem(QString ("%1").arg(deviceProp->warpSize),8,1);
    addItem("Maximum number of threads per multiprocessor:",9,0);
    addItem(QString ("%1").arg(deviceProp->maxThreadsPerMultiProcessor),9,1);
    addItem("Maximum number of threads per block:",10,0);
    addItem(QString ("%1").arg(deviceProp->maxThreadsPerBlock),10,1);
    addItem("Max dimension size of a thread block (x,y,z):",11,0);
    addItem(QString ("(%1, %2, %3)").arg(deviceProp->maxThreadsDim[0]).arg(  deviceProp->maxThreadsDim[1]).arg(  deviceProp->maxThreadsDim[2]),11,1);
    addItem("Max dimension size of a grid size    (x,y,z):",12,0);
    addItem(QString ("(%1, %2, %3)\n").arg(deviceProp->maxGridSize[0]).arg(deviceProp->maxGridSize[1]).arg(deviceProp->maxGridSize[2]),12,1);
    addItem("Run time limit on kernels: ",13,0);
    addItem(QString ("%1\n").arg(deviceProp->kernelExecTimeoutEnabled ? "Yes" : "No"),13,1);
    addItem("Integrated GPU sharing Host Memory: ",14,0);
    addItem( QString ("%1\n").arg(deviceProp->integrated ? "Yes" : "No"),14,1);

    ui->tableWidget->resizeColumnsToContents();
    ui->tableWidget->resizeRowsToContents();
}

void DialogSelectHardware::setDevice()
{
    cudaSetDevice(ui->deviceComboBox->currentIndex());
    selectDevice=ui->deviceComboBox->currentIndex();
}
