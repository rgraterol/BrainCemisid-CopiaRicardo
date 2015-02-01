#-------------------------------------------------
#
# Project created by QtCreator 2013-05-17T12:45:15
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = BrainCemisidJJ
TEMPLATE = app


SOURCES += Main/main.cpp\
        Main/mainwindow.cpp \
    Graphics/chemicalLayer.cpp \
    Graphics/formteaching.cpp \
    Class/interface.cpp \
    Graphics/canvas.cpp \
    Graphics/viewfinder.cpp \
    Graphics/dialogselecthardware.cpp \
    Graphics/dialogstatistics.cpp \
    Graphics/dialogtablebns.cpp \
    Graphics/dialogconsultneuron.cpp \
    Class/sum_queue.cpp

HEADERS  += Main/mainwindow.h \
    Graphics/chemicalLayer.h \
    Class/neuralNetwork.h \
    CUDA/timer.h \
    CUDA/helper_string.h \
    CUDA/helper_cuda.h \
    CUDA/utilCuda.h \
    Graphics/formteaching.h \
    Class/interface.h \
    CUDA/lock.h \
    Graphics/canvas.h \
    Graphics/viewfinder.h \
    Class/statistic.h \
    Graphics/dialogselecthardware.h \
    Graphics/dialogconsultneuron.h \
    Graphics/dialogstatistics.h \
    Graphics/dialogtablebns.h \
    Class/relationNetwork.h \
    Class/countNetwork.h \
    Class/culturalNet.h \
    Class/sumNetwork.h \
    Class/sumQueue.h

FORMS    += Forms/mainwindow.ui \
    Forms/formteaching.ui \
    Forms/dialogselecthardware.ui \
    Forms/dialogconsultneuron.ui \
    Forms/dialogstatistics.ui \
    Forms/dialogtablebns.ui

RESOURCES += \
    imageResource.qrc

# Directorios Generar proyecto
DESTDIR     =  $$system(pwd)
OBJECTS_DIR =  $$DESTDIR/Obj
# + Banderas de C +
QMAKE_CXXFLAGS_RELEASE =-O3

# Cuda
CUDA_SOURCES += CUDA/cuda_code.cu

# Ruta al cuda toolkit instalación
CUDA_DIR       = /usr/local/cuda-5.5
# Ruta a los archivos de cabecera y bibliotecas
INCLUDEPATH   +=  $$CUDA_DIR/include
QMAKE_LIBDIR  +=  $$CUDA_DIR/lib64     # Tenga en cuenta que estoy usando un sistema operativo de 64 bits de
# Librerías utilizadas en su código
LIBS +=   -lcuda -lcudart
# GPU arquitectura
CUDA_ARCH      =  sm_21                 # Yeah! Tengo un nuevo dispositivo. Ajuste con su capacidad de cálculo
# Aquí hay algunas banderas NVCC Siempre he usado por defecto.
NVCCFLAGS      =--compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v



# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
OTHER_FILES += CUDA/cuda_code.cu
