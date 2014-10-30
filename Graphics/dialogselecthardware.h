#ifndef DIALOGSELECTHARDWARE_H
#define DIALOGSELECTHARDWARE_H

#include <QDialog>
#include <QMessageBox>
#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDA/helper_string.h>
#include <CUDA/helper_cuda.h>
#include <iostream>

namespace Ui {
class DialogSelectHardware;
}

class DialogSelectHardware : public QDialog
{
    Q_OBJECT
public slots:
    void ChangeText(int indexDevice);
    void setDevice ();
public:
    explicit DialogSelectHardware(QWidget *parent = 0, int selectDevice =-1);
    void  setListDevice();
    void  setVisibleSelectButton(bool active);
    void  showWarning(QString windowTitle,QString message);
    void  addItem(QString text,int row,int col);
    int   getIndexSelectecDevice();
    ~DialogSelectHardware();
    
private:
    Ui::DialogSelectHardware *ui;
    cudaDeviceProp * deviceProp;
    int deviceCount;
    int selectDevice;
};

#endif // DIALOGSELECTHARDWARE_H
