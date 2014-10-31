#include "mainwindow.h"
#include "ui_mainwindow.h"

//PRUEBA
//prototyping methods that invoke cuda kernels
extern "C"
void boot(NeuralNetwork * & neuralSenses,const SizeNet & sizeNet, Statistic * & statistic);

extern "C"
stateNeuralNetwork recognize(NeuralNetwork *  neuralSenses,const SizeNet & sizeNet,
                             unsigned char * h_pattern , Interface * interface, Statistic * & statistic);
extern "C"
void correct(NeuralNetwork * neuralSenses , const SizeNet & sizeNet,
             unsigned char   desiredOutput, int maxThreadsPerBlock ,Statistic * &statistic);
extern "C"
void reset(NeuralNetwork * neuralSenses , const SizeNet & sizeNet,int maxThreadsPerBlock,Statistic *& statistic);


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    try
    {
        initGui();
        setNull();
        showSelectDevice();
        intitializeSenses();
        generateVectorsCharacteristic();
        createTablesCharacteristic();
        createInterfacesTeaching();
        earTraining();
    }
    catch (bad_alloc & exceptionMem)
    {
        QString text(exceptionMem.what());
        showWarning("Fallo de memoria",
                    "Su sistema no posee memoria RAM suficiente para ejecutar aplicacion ERROR: "
                    +text);
        freeUi();
        freeMem();
        exit(EXIT_SUCCESS);
    }
}

MainWindow::~MainWindow()
{
    freeUi();
    freeMem();
}

void MainWindow::showSelectDevice(bool isVisibleButton)
{
    if(!multipleDevice() && isVisibleButton)
        return;

    dialogSelectHardware = new DialogSelectHardware(0,selectedDevice);
    dialogSelectHardware->setVisibleSelectButton(isVisibleButton);
    dialogSelectHardware->exec();

    if(isVisibleButton){
        selectedDevice = dialogSelectHardware->getIndexSelectecDevice();
    }
    delete dialogSelectHardware;
    dialogSelectHardware = NULL;
}

void MainWindow::freeVectorsCharacteristic()
{
    freeGenericPtr(characteristicVectorEar);
    freeGenericPtr(characteristicVectorEye);
}

void MainWindow::freeFormTeaching()
{
    if(formsTeaching != NULL)
    {
        for (int i = 0; i < numSenses; i++)
            delete formsTeaching[i];

        delete formsTeaching;
    }

    freeGenericPtr(isInactivateSense);
}

void MainWindow::freeStates()
{
    freeGenericPtr(stateSenses);
}

void MainWindow::freeSenses()
{
    if(neuralSenses != NULL)
    {
        for (register int i = 0; i < this->numSenses; i++) {
            freeGenericPtr( neuralSenses[i].vectorNeuron);
            freeGenericPtr( neuralSenses[i].vectorFlags );
            freeGenericPtr( neuralSenses[i].binaryCharacteristic );
        }

        delete neuralSenses;
    }
}

void MainWindow::freeInterface()
{
    if(interface!= NULL)
    {
        interface[HEARING].freeMem(true);
        interface[SIGHT].freeMem(true);
        delete interface;
    }
}

void MainWindow::freeUi()
{
    delete ui;
    delete chemicalLayerEar;
    delete chemicalLayerEye;
    freeGenericPtr(image);
    freeGenericPtr(dialogConsult);
    freeGenericPtr(dialogStatistics);
    freeGenericPtr(dialogTable);
    freeGenericPtr(dialogSelectHardware);
}

void MainWindow::freeMem()
{
    freeFormTeaching();
    freeStates();
    freeVectorsCharacteristic();
    freeSenses();
    freeInterface();
    freeGenericPtr(statistics);
}

void MainWindow::generateVectorsCharacteristic()
{
    characteristicVectorEar = chemicalLayerEar->generateCharacteristic();
    characteristicVectorEye = chemicalLayerEye->generateCharacteristic();
}


void MainWindow::processGrid()
{
    clearTables();
    //printCountNetwork();

    if(!chemicalLayerEye->getNoData())
    {
        try {

           /* orderNetwork->countNet[orderNeuron].vectorNetworkCount[kNeuron]=1;
            orderNetwork->countNet[orderNeuron].vectorPointerCount[kNeuron]=0;
            QString cN = QString::number(kNeuron);
            QString oN = QString::number(orderNeuron);
            ui->textBrowser->setText(cN+" "+oN);
            kNeuron=0;
            orderNeuron++;*/

            stateSenses  [SIGHT]   = recognize(&neuralSenses[SIGHT]  ,sizeNet,characteristicVectorEye,&interface[SIGHT],statistics);
        } catch (string text) {
            QMessageBox::critical(this,"CUDA ERROR",QString(text.c_str()).append("\n Se cerrara Aplicación"));
            exit(EXIT_FAILURE);
        }

        formsTeaching[SIGHT]->setState    ( stateSenses [SIGHT]  ,interface[SIGHT].arrayCategory[0]  );
    }

    else
        isInactivateSense   [SIGHT]   = true;

    if(!chemicalLayerEar->getNoData())
    {
        try {
            stateSenses  [HEARING] = recognize(&neuralSenses[HEARING],sizeNet,characteristicVectorEar,&interface[HEARING],statistics);
           // printf("%d\n", characteristicVectorEar);

            if( stateSenses  [HEARING] == IS_HIT && !isInactivateSense[SIGHT])
                emit cross();

        } catch (string text) {
            QMessageBox::critical(this,"CUDA ERROR",QString(text.c_str()).append("\n Se cerrara Aplicación"));
            exit(EXIT_FAILURE);
        }

        formsTeaching[HEARING]->setState( stateSenses [HEARING],interface[HEARING].arrayCategory[0]);
    }

    else
        isInactivateSense   [HEARING] = true;

}

void MainWindow::clearTables()
{
    generateVectorsCharacteristic();
    createTablesCharacteristic();
}

void MainWindow::learnHearing()
{
    learn(HEARING);
}

void MainWindow::learnSight()
{
    learn(SIGHT);
}

void MainWindow::resetHearing()
{
    try {
        reset(&neuralSenses[HEARING],sizeNet,deviceProp.maxThreadsPerBlock,statistics);
    } catch (string text) {
        QMessageBox::critical(this,"CUDA ERROR",QString(text.c_str()).append("\n Se cerrara Aplicación"));
        exit(EXIT_FAILURE);
    }
}

void MainWindow::resetSight()
{
    try {
        reset(&neuralSenses[SIGHT],sizeNet,deviceProp.maxThreadsPerBlock,statistics);
    } catch (string text) {
        QMessageBox::critical(this,"CUDA ERROR",QString(text.c_str()).append("\n Se cerrara Aplicación"));
        exit(EXIT_FAILURE);
    }
}

void MainWindow::activateButtonBip()
{
    if(!(chemicalLayerEar->getNoData()) || !(chemicalLayerEye->getNoData())) {
        ui->pushButtonBip->setEnabled(true);
    }

    else {
        ui->pushButtonBip->setEnabled(false);
    }
}

void MainWindow::activateButtonProcess()
{
    if(! ( chemicalLayerEar->getNoData() ) || !( chemicalLayerEye->getNoData() ) )
        ui->pushButtonProcess->setEnabled(true);

    else
        ui->pushButtonProcess->setEnabled(false);
}

void MainWindow::activeLayers(bool active)
{
    if(active)
    {
        if(isInactivateSense[SIGHT] == true &&  isInactivateSense[HEARING] == true)
        {
            ui->lineEditEarInput->clear();
            chemicalLayerEye->clear();
            chemicalLayerEar->clear();
            clearTables();
            setFormsCheck();
            activateInterface(active);
        }
    }
    else
        activateInterface(active);

    ui->pushButtonProcess->setEnabled(false);
    //ui->pushButtonBip->setEnabled(false);
}

void MainWindow::finishGoodAnswer(senses sense)
{
    isInactivateSense[sense] = true;
    activeLayers(true);
}

void MainWindow::paintNetNeuron(senses sense, bool onlyHits)
{
    QString nameFile= ( sense == HEARING ) ? "./Obj/net_hearing.dot" : "./Obj/net_sight.dot";
    generateDot(nameFile,sense,onlyHits);
    generatePng(nameFile);
    ViewFinder &view = ViewFinder::getInstance(this);
    view.showNet();
}

void MainWindow::paintBinaryCharacteristic(senses sense, int ptr)
{
    unsigned short  displacement = 8 * sizeof (unsigned short) -1;
    unsigned short  mask=  1 << displacement;
    freeGenericPtr(image);

    image = new QImage(QSize(400,400), QImage::Format_MonoLSB);
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
    ViewFinder &view = ViewFinder::getInstance(this);
    view.showBinaryCharacteristic(image);
}

void MainWindow::paintStateHearing()
{
    paintNetNeuron(HEARING);
}

void MainWindow::paintStateSight()
{
    paintNetNeuron(SIGHT);
}

void MainWindow::paintHearing()
{
    if(*(neuralSenses[HEARING].ptr) == 0)
        showWarning("Sentido Tabula Rasa","Bloque neuronal(BNS) del oido se encuentra tabula Rasa");

    else
        paintNetNeuron(HEARING,false);
}

void MainWindow::paintSight()
{
    if(*(neuralSenses[SIGHT].ptr) == 0)
        showWarning("Sentido Tabula Rasa","Bloque neuronal(BNS) de la vista se encuentra tabula Rasa");

    else
        paintNetNeuron(SIGHT,false);
}

void MainWindow::activateMainWindow(bool activate)
{
    this->setEnabled(activate);
}

void MainWindow::showDialogConsult()
{
    if( (*(neuralSenses[HEARING].ptr) == 0 ) && (*(neuralSenses[SIGHT].ptr) == 0 ))
    {
        showWarning("tabula Rasa","No existen neuronas con conocimiento");
        return;
    }

    dialogConsult = new DialogConsultNeuron(neuralSenses[HEARING].ptr,neuralSenses[SIGHT].ptr);

    if(dialogConsult->exec()==QDialog::Accepted)
        paintBinaryCharacteristic(dialogConsult->radioButtonActive(),dialogConsult->returnIdNeuron());

    freeGenericPtr(dialogConsult);
    dialogConsult = NULL;
}

void MainWindow::showDialogStatistic()
{
    dialogStatistics = new DialogStatistics();
    dialogStatistics->setItems(statistics);
    dialogStatistics->exec();
    freeGenericPtr(dialogStatistics);
    dialogStatistics = NULL;
}

void MainWindow::showDialogTableNeuron()
{
    dialogTable= new DialogTableBNS(0,neuralSenses,&sizeNet);
    dialogTable->setWindowModality(Qt::WindowModal);
    this->setVisible(false);

    if( dialogTable->exec()==QDialog::Rejected)
        this->setVisible(true);

    freeGenericPtr(dialogTable);
    dialogTable = NULL;
}

void MainWindow::aboutBrainCemisid()
{
    QString title = "<H2> Proyecto Brain Cemisid J&J </H2>";
    QString version = "<H5> Versión 1.0 <br> svn checkout https://braincemisidjj.googlecode.com/svn/trunk/ </H5> <br>";
    QString description;
    description.append("<b>Proyecto de construcción de un cerebro usando arquitectura por capas, ");
    description.append("esta Aplicación es la Creación del Estado Cerebral:</b> <br>");
    description.append("<ol><li>Mecanismo de captura de estímulos (sensores)</li>");
    description.append("<li>Proceso de transformación de esa captura en una   característica (vector 32 pos)</li>");
    description.append("<li>Creación de Bloques Neuronales Sensoriales(BNS) usando   Redes Neuronales de base radial desarrolladas en CEMISID(Liliana-Carlos)</li>");
    description.append("<li>interfaz Gráfica que muestre el funcionamiento<br></li></ol>");
    QString titleProgramer = "<b>Desarrollado por:</b> <br>";
    QString programer = "Jonathan Monsalve <br><br>";
    QString titleEmail = "<b>Dirección electronica:</b><br>";
    QString email = "j.jmonsalveg@gmail.com<br><br>";
    QString titleTutor = "<b>Bajo la Tutoría:</b> <br>";
    QString tutor = "Dr. Gerard Páez Monzón <br><br>";
    QString university = "Universidad de los Andes, 2013.";


    QMessageBox::about(this, tr("Acerca de Proyecto Brain Cemisid J&J V1.0"), "<p align=\"center\">" + title +
                       version +"</p>"+ "<p align=\"justify\">"+description +"</p>"+"<p align=\"center\">"+ titleProgramer + programer +
                       titleEmail + email + titleTutor + tutor + "<font color = #FF8000>" + university +
                       "</font>" + "</p>");
}

void MainWindow::aboutCuda()
{
    QString title = "<h1><font color=#76B900>PLATAFORMA DE CÁLCULO PARALELO CUDA</font></h1>";
    QString version = "<H5> Version 5.5(para el desarrollo de esta aplicacion) </H5> <br>";
    QString p1= "<p>La plataforma de cálculo paralelo "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"http://developer.nvidia.com/cuda\" onclick=\"s_objectID=\"http://developer.nvidia.com/cuda_1\";return this.s_oc?this.s_oc(e):true\">CUDA®</a> "//
            "proporciona unas cuantas extensiones de C y C++ que permiten implementar el paralelismo en el procesamiento de tareas y datos con diferentes niveles de granularidad. El programador puede expresar ese paralelismo mediante diferentes lenguajes de alto nivel como C, "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"http://developer.nvidia.com/cuda-toolkit\" onclick=\"s_objectID=\"http://developer.nvidia.com/cuda-toolkit_1\";return this.s_oc?this.s_oc(e):true\">C++</a> "//
            "y Fortran o mediante estándares abiertos como las directivas de OpenACC"//
            ". En la actualidad, la plataforma CUDA se utiliza en miles de "//
            "aplicaciones aceleradas en la GPU y en miles de "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"http://scholar.google.com/scholar?q=cuda+gpu\" onclick=\"s_objectID=\"http://scholar.google.com/scholar?q=cuda+gpu_1\";return this.s_oc?this.s_oc(e):true\">artículos de investigación publicados</a> "//
            ".<br>";
    QString p2 = "<p>Los desarrolladores disponen de una gama completa de "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"http://developer.nvidia.com/cuda-tools-ecosystem\" onclick=\"s_objectID=\"http://developer.nvidia.com/cuda-tools-ecosystem_1\";return this.s_oc?this.s_oc(e):true\">herramientas y soluciones pertenecientes al ecosistema de CUDA</a> "//
            ". Visita "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"http://developer.nvidia.com/cuda\" onclick=\"s_objectID=\"http://developer.nvidia.com/cuda_2\";return this.s_oc?this.s_oc(e):true\">CUDA Zone</a> "//
            "para obtener más información sobre el desarrollo con CUDA."//
            "</p><br>";
    QString p3 = "<p>Si quieres empezar a utilizar el "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"https://developer.nvidia.com/get-started-parallel-computing\" onclick=\"s_objectID=\"https://developer.nvidia.com/get-started-parallel-computing_1\";return this.s_oc?this.s_oc(e):true\">procesamiento paralelo</a> "//
            "o "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"https://developer.nvidia.com/cuda-downloads\" onclick=\"s_objectID=\"https://developer.nvidia.com/cuda-downloads_1\";return this.s_oc?this.s_oc(e):true\">descargar el último software de CUDA</a> "//
            "y necesitas información, entra en "//
            "<a rel=\"nofollow\" target=\"_blank\" href=\"http://developer.nvidia.com/cuda\" onclick=\"s_objectID=\"http://developer.nvidia.com/cuda_3\";return this.s_oc?this.s_oc(e):true\">CUDA Developer Zone</a> "//
            ".</p><br>";

    QMessageBox messageAbout;
    messageAbout.setWindowTitle (tr("Acerca de CUDA"));
    messageAbout.setIconPixmap(QPixmap("icons/nvidia-logo.jpg"));
    messageAbout.setText ("<p align=\"center\">" + title +
                          version +"</p>"+ "<p align=\"justify\">"+p1 +"</p>"+"<p align=\"center\">"+ p2 + p3 +
                          "</p>");
    messageAbout.exec ();

}

void MainWindow::launchWave()
{
    chemicalLayerEar->clear();

    if(ui->lineEditEarInput->text().isEmpty())
        return;

    int categoryWave = returnCategory(ui->lineEditEarInput->text());
    int temp [16];

    for (int indexNeuron = 0; indexNeuron < *(neuralSenses[HEARING].ptr); indexNeuron++) {

        int currentCategory = neuralSenses[HEARING].vectorFlags[indexNeuron * SIZE_FLAGS + CAT];

        if(currentCategory != categoryWave)
            continue;

        for (int i = 0; i < SIZE_CHARACTERISTIC/2 ; i++)
            temp[i]=neuralSenses[HEARING].binaryCharacteristic[indexNeuron * SIZE_CHARACTERISTIC/2 + i];

        chemicalLayerEar->paintPattern(temp,SIZE_CHARACTERISTIC/2);
        break;
    }
}

void MainWindow::runCrossing()
{
    formsTeaching[SIGHT]->getLineEditInput()->setText(QString::number(interface[HEARING].id[0]));
    formsTeaching[SIGHT]->getPtrButtonTeach()->setEnabled(true);
}

void MainWindow::generateDot(QString nameFile, senses sense, bool onlyHits)
{
    ofstream file(nameFile.toStdString().c_str());
    int id, category,centinel;

    unsigned char * vector= (onlyHits) ?  interface[sense].arrayCategory : returnVectorCategory(sense) ;
    centinel= (onlyHits) ?*(interface[sense].hits) : *(neuralSenses[sense].ptr);

    file<<"graph net_neuron{\n";
    file<<"rankdir=LR;\n";
    for(int j=0; j <= 12;j++){
        file<<"subgraph cluster_"<<j<<"{ ";

        for (int i = 0; i < centinel ; i++) {

            id=(onlyHits) ? interface[sense].id[i] : i;
            category=vector[i];

            if(category != j){

                if(j==11 && category==43)
                    file<<"\"item"<<id<<"\" [label=  \"id neurona = "<<id<<"\\nCategoria = '+'\"];\n";
                if(j==12 && category==61)
                    file<<"\"item"<<id<<"\" [label=  \"id neurona = "<<id<<"\\nCategoria = '='\"];\n";
            }

            else
                file<<"\"item"<<id<<"\" [label=  \"id neurona = "<<id<<"\\nCategoria= "<<category<<"\"];\n";

        }

        file<<"}\n";
    }

    file<<"\n}";
    file.close();

    if(!onlyHits)
        delete vector;
}

unsigned char *MainWindow::returnVectorCategory(senses sense)
{
    int ptr = *(neuralSenses[sense].ptr);
    unsigned char * vector= new unsigned char [ptr];

    for(register int i=0 ; i < ptr ; i++)
        vector[i] = neuralSenses[sense].vectorFlags[i * 9 + CAT];

    return vector;
}

bool MainWindow::multipleDevice()
{
    int deviceCount;

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        QString text("ERROR: cudaGetDeviceCount returned "+QString::number((int)error_id).append(cudaGetErrorString(error_id)));
        showWarning("Device no encontrado",text);
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        showWarning("No Compatibilidad","ERROR: There are no available device(s) that support CUDA\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount==1)
    {
        cudaSetDevice(0);
        selectedDevice = 0;
        return false;
    }

    return true;

}

void MainWindow::generatePng(QString nameFile)
{
    QString program = "dot";
    QStringList arguments;
    //dot net_hearing.dot  -o net.svg -Tsvg
    arguments <<nameFile.toStdString().c_str()<<"-o" << "./Obj/net.svg"<<"-Tsvg";

    QProcess *myProcess = new QProcess(this);
    myProcess->start(program, arguments);
}

void MainWindow::showWarning(QString windowTitle, QString message)
{
    QMessageBox::warning(0,windowTitle,message,QMessageBox::Ok);
}


void MainWindow::learn(senses sense)
{
    isInactivateSense[sense] = true;

    switch (stateSenses[sense]) {
    case NO_HIT:
        realLearn(sense);
        break;

    case IS_HIT:
        try{
        correct(&neuralSenses[sense],sizeNet,formsTeaching[sense]->getLineEditInput()->text().toInt(),deviceProp.maxThreadsPerBlock,statistics);
        realLearn(sense);
        reset(&neuralSenses[sense],sizeNet,deviceProp.maxThreadsPerBlock,statistics);
        break;
    } catch (string text) {
            QMessageBox::critical(this,"CUDA ERROR",QString(text.c_str()).append("\n Se cerrara Aplicación"));
            exit(EXIT_FAILURE);
        }

    case DIFF:
        try{
        correct(&neuralSenses[sense],sizeNet,formsTeaching[sense]->getLineEditInput()->text().toInt(),deviceProp.maxThreadsPerBlock,statistics);
        stateSenses[sense] = recognize(&neuralSenses[sense],sizeNet,characteristicVectorEar,&interface[sense],statistics);
        break;
    } catch (string text) {
            QMessageBox::critical(this,"CUDA ERROR",QString(text.c_str()).append("\n Se cerrara Aplicación"));
            exit(EXIT_FAILURE);
        }

    default:
        exit(EXIT_FAILURE);
    }

    formsTeaching[sense]->init();
    activeLayers(true);

}

void MainWindow::realLearn(senses sense)
{
    unsigned char ptr =( *(neuralSenses[sense].ptr) )++ ;

    if(ptr > sizeNet.numNeuron){
        showWarning("Red no dimensionable","No es posible aprender mas Patrones\nERROR:Numero de neuronas Agotadas");
        return;
    }

    neuralSenses[sense].vectorFlags[ptr * SIZE_FLAGS + KNW ] = 1;
    neuralSenses[sense].vectorFlags[ptr * SIZE_FLAGS + CAT ] = returnCategory(formsTeaching[sense]->getLineEditInput()->text());

    learnBinaryCharacteristic(sense, ptr);
}

unsigned char MainWindow::returnCategory(QString cad)
{
    if(cad != "=" && cad != "+" )
        return cad.toInt();

    if (cad == "=")
        return '=';

    if(cad == "+")
        return'+';

    exit(1);
}

void MainWindow::learnBinaryCharacteristic(senses sense,int ptr)
{
    characteristicBinaryVector = (sense == SIGHT) ? chemicalLayerEye->generateBinaryCharacteristic():
                                                    chemicalLayerEar->generateBinaryCharacteristic();
    for (int i = 0; i < 16; i++)
        neuralSenses[sense].binaryCharacteristic[ptr * 16+ i] = characteristicBinaryVector [i];
}

void MainWindow::initGui()
{
    ui->setupUi(this);
    chemicalLayerEar= new ChemicalLayer(16,16,10,10);
    chemicalLayerEar->setIsEditable(false);
    chemicalLayerEye= new ChemicalLayer(16,16,10,10);
    ui->horizontalLayoutEar->addWidget(chemicalLayerEar);
    ui->horizontalLayoutEye->addWidget(chemicalLayerEye);
    ui->pushButtonProcess->setEnabled(false);
    ui->pushButtonBip->setEnabled(false);
    ui->textBrowser->hide();
    ui->pushButton_stopCount->hide();
    ui->pushButton_stopCount->hide();
    ui->pushButton_teachBip->hide();
    ui->pushButton_teachClack->hide();
    QRegExp alphabet("(\\d|=|\\+)");
    ui->lineEditEarInput->setValidator(new QRegExpValidator(alphabet,this));

    ViewFinder::getInstance(this);

    connect(chemicalLayerEar     ,SIGNAL(change())   ,this,SLOT(activateButtonProcess()));
    connect(chemicalLayerEar    ,SIGNAL(change())   ,this,SLOT(activateButtonBip()));
    connect(chemicalLayerEye     ,SIGNAL(change())   ,this,SLOT(activateButtonProcess()));
    connect(chemicalLayerEye    ,SIGNAL(change())   ,this,SLOT(activateButtonBip()));
    connect(ui->pushButtonProcess,SIGNAL(clicked())  ,this,SLOT(activeLayers()) );
    connect(ui->buttonClearEar   ,SIGNAL(clicked())  ,chemicalLayerEar,SLOT(clear()));
    connect(ui->buttonClearEye   ,SIGNAL(clicked())  ,chemicalLayerEye,SLOT(clear()));
    connect(ui->actionQt         ,SIGNAL(triggered()),qApp,SLOT(aboutQt()));
    connect(ui->lineEditEarInput ,SIGNAL(textChanged(QString)),this,SLOT(launchWave()));
    //connect(ui->pushButtonBip ,SIGNAL(clicked())  ,this,SLOT(countProtocol()));
    connect(ui->pushButton_startCount ,SIGNAL(clicked()), this, SLOT(startCount()));
    connect(ui->pushButton_stopCount, SIGNAL(clicked()), this, SLOT(stopCount()));
    connect(ui->pushButton_teachBip ,SIGNAL(clicked())  ,this,SLOT(countProtocol()));
    connect(ui->pushButton_teachClack, SIGNAL(clicked()), this, SLOT(orderProtocol()));
    connect(this,SIGNAL(cross()),this,SLOT(runCrossing()));
    setMinimumSize(1000,800);
    setMaximumSize(1000,800);

}


void MainWindow::activateInterface(bool state)
{
    ui->lineEditEarInput->setEnabled(state);
    ui->buttonClearEar->setEnabled(state);
    ui->buttonClearEye->setEnabled(state);
    chemicalLayerEar->setActiveLayer(state);
    chemicalLayerEye->setActiveLayer(state);
}



void MainWindow::setNull()
{
    characteristicVectorEar = NULL;
    characteristicVectorEye = NULL;
    characteristicBinaryVector = NULL;
    neuralSenses         = NULL;
    stateSenses          = NULL;
    formsTeaching        = NULL;
    interface            = NULL;
    isInactivateSense      = NULL;
    image                = NULL;
    dialogConsult        = NULL;
    dialogStatistics     = NULL;
    dialogTable          = NULL;
    dialogSelectHardware = NULL;
    statistics           = NULL;
    selectedDevice       = -1;
}

void MainWindow::createTablesCharacteristic()
{
    initializeTable();
    createStringForTable();
}

void MainWindow::initializeTable()
{
    ui->tableWidgetColumnEar->setRowCount(1);
    ui->tableWidgetColumnEar->setColumnCount(16);
    ui->tableWidgetColumnEar->verticalHeader()->hide();

    ui->tableWidgetRowEar->setRowCount(16);
    ui->tableWidgetRowEar->setColumnCount(1);
    ui->tableWidgetRowEar->horizontalHeader()->hide();

    ui->tableWidgetColumnEye->setRowCount(1);
    ui->tableWidgetColumnEye->setColumnCount(16);
    ui->tableWidgetColumnEye->verticalHeader()->hide();

    ui->tableWidgetRowEye->setRowCount(16);
    ui->tableWidgetRowEye->setColumnCount(1);
    ui->tableWidgetRowEye->horizontalHeader()->hide();
}


/*===============================INICIALIZANDO NEURONAS DE CONTEO Y ORDEN*========================================*/

int kNeuron =1;
int orderNeuron =1;

/*===================================================================================================================*/

void MainWindow::intitializeSenses(int numSenses)
{
    int calc;
    orderNeuron=1;
    kNeuron=1;
    this->numSenses=numSenses;
    int numNeuron=calculateNumNeuron();

    statistics   = new Statistic [4];
    neuralSenses = new NeuralNetwork[numSenses];
    stateSenses  = new stateNeuralNetwork[numSenses];
    interface    = new Interface [numSenses];
    sizeNet.numNeuron=numNeuron;


    orderNetwork = new OrderNetwork;

    calc = sizeNet.numNeuron * sizeof(unsigned char);
    sizeNet.sizeVectorNeuron         = calc * 32;
    sizeNet.sizevectorFlags          = calc * 9;
    sizeNet.sizeBinaryCharacteristic = sizeNet.numNeuron * sizeof(unsigned short) * 16;

    orderNetwork->countNet = new CountNetwork [sizeNet.numNeuron];

    countNetwork = new CountNetwork;
    countNetwork->vectorNetworkCount = new unsigned char [sizeNet.numNeuron * 32];
    countNetwork->vectorPointerCount = new unsigned char [sizeNet.numNeuron * 9];
    countNetwork->bipPointer = new unsigned char [sizeNet.numNeuron * 9];
    countNetwork->clackPointer = new unsigned char [sizeNet.numNeuron * 9];
    countNetwork->ptr = new unsigned char (1);

    orderNetwork->bumPointer = new unsigned char [sizeNet.numNeuron * 9];
    orderNetwork->category = new unsigned char [sizeNet.numNeuron * 32];
    orderNetwork->order = new unsigned char [sizeNet.numNeuron * 32];

    for(register int i=0; i<sizeNet.numNeuron; i++) {
        orderNetwork->countNet[i].vectorNetworkCount = new unsigned char [sizeNet.numNeuron * 32];
        orderNetwork->countNet[i].vectorPointerCount = new unsigned char [sizeNet.numNeuron * 9];
        orderNetwork->countNet[i].ptr = new unsigned char (1);
    }

    for (register int i = 0; i < numSenses; i++) {
        neuralSenses[i].vectorNeuron         = new unsigned char [ sizeNet.numNeuron * 32 ];
        neuralSenses[i].vectorFlags          = new unsigned char [ sizeNet.numNeuron * 9  ];
        neuralSenses[i].binaryCharacteristic = new unsigned short [ sizeNet.numNeuron * 16 ];
        neuralSenses[i].ptr                  = new unsigned char (0);
    }
    try {
        boot(neuralSenses,sizeNet,statistics);
    } catch (string text) {
        QMessageBox::critical(this,"CUDA ERROR",QString(text.c_str()).append("\n Se cerrara Aplicación"));
        exit(EXIT_FAILURE);
    }
}

void MainWindow::createStringForTable()
{
    QStringList listStringRow,listStringColumn;
    QString auxRow,auxColumn;

    for (int i = 0; i < 16; i++) {
        auxRow="v[";
        auxRow.append(QString::number(i+1));
        auxRow.append("]");
        listStringRow<<auxRow;

        auxColumn="v[";
        auxColumn.append(QString::number(i+17));
        auxColumn.append("]");
        listStringColumn<<auxColumn;


        QTableWidgetItem * newItem = new QTableWidgetItem(QString::number(characteristicVectorEar[i]));
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        ui->tableWidgetRowEar->setItem(i, 0, newItem);

        newItem = new QTableWidgetItem(QString::number(characteristicVectorEar[i+16]));
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        ui->tableWidgetColumnEar->setItem(0, i, newItem);

        newItem = new QTableWidgetItem(QString::number(characteristicVectorEye[i]));
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        ui->tableWidgetRowEye->setItem(i, 0, newItem);

        newItem = new QTableWidgetItem(QString::number(characteristicVectorEye[i+16]));
        newItem->setTextAlignment(Qt::AlignVCenter);
        newItem->setTextAlignment(Qt::AlignHCenter);
        ui->tableWidgetColumnEye->setItem(0, i, newItem);
    }

    setDataTable(listStringRow,listStringColumn);
}

void MainWindow::earTraining()
{
    senseTraining("ear_category.o","ear_wave.o","./Obj/",12,HEARING);
}

void MainWindow::senseTraining( QString nameFileCategories, QString nameFileWaves,QString path,int numPatterns,senses sense)
{
    int category;
    int waveCharacteristic[SIZE_CHARACTERISTIC/2];
    bool condition;
    ChemicalLayer * layer =( sense == HEARING)? chemicalLayerEar: chemicalLayerEye;
    FormTeaching  * formTeaching= formsTeaching[sense];

    ifstream categories((path+nameFileCategories).toStdString().c_str());
    if(!categories){
        showWarning("Error abriendo Archivo","Archivo que contiene categorías para entrenar oido no se puede abrir aplicacion se cerrara");
        exit(EXIT_SUCCESS);
    }

    ifstream waves((path+nameFileWaves).toStdString().c_str());
    if(!waves){
        showWarning("Error abriendo Archivo","Archivo que contiene Ondas para entrenar oido no se puede abrir aplicacion se cerrara");
        categories.close();
        exit(EXIT_SUCCESS);
    }

    do {

        categories.seekg(0);
        waves.seekg(0);
        condition = false;
        for (int i = 0;  i < numPatterns; i++) {
            //interaction is simulated user-GUI
            waves.read     (reinterpret_cast<char*>(&waveCharacteristic),SIZE_CHARACTERISTIC/2 *sizeof(int));
            categories.read(reinterpret_cast<char*>(&category),sizeof(int));
            layer->paintPattern(waveCharacteristic , SIZE_CHARACTERISTIC/2);

            processGrid();
            activeLayers();

            switch (stateSenses[sense]) {

            case NO_HIT:
                formTeaching->getLineEditInput()->setText(QString::number(category));
                formTeaching->getPtrButtonTeach()->clicked();
                break;

            case IS_HIT:
                if(returnCategory(formTeaching->getLineEditOut()->text()) != category)
                {
                    formTeaching->badAnswer();
                    formTeaching->getLineEditInput()->setText(QString::number(category));
                    formTeaching->getPtrButtonTeach()->clicked();
                    condition = true;
                }
                else
                {
                    formTeaching->init();
                    finishGoodAnswer(sense);
                    ( sense == HEARING)? resetHearing(): resetSight();
                }
                break;


            case DIFF:

                formTeaching->getLineEditInput()->setText(QString::number(category));
                formTeaching->getPtrButtonTeach()->clicked();
                condition = true;
                break;

            default:
                exit(EXIT_FAILURE);
            }
        }

    } while (condition);

    categories.close();
    waves.close();
}

void MainWindow::setDataTable(QStringList listStringRow, QStringList listStringColumn)
{
    ui->tableWidgetColumnEar->setHorizontalHeaderLabels(listStringColumn);
    ui->tableWidgetRowEar->setVerticalHeaderLabels(listStringRow);

    ui->tableWidgetColumnEye->setHorizontalHeaderLabels(listStringColumn);
    ui->tableWidgetRowEye->setVerticalHeaderLabels(listStringRow);
}

int MainWindow::calculateNumNeuron()
{
    cudaSetDevice(selectedDevice);
    cudaGetDeviceProperties(&deviceProp, selectedDevice);

    int sizeVectorCharacteristic= sizeof(unsigned char)* 32;
    int sizeVectorFlags= sizeof(unsigned char)* 9;
    int sizePtr= sizeof(unsigned char);
    int sizeNeuron= sizeVectorCharacteristic + sizeVectorFlags + sizePtr;
    int numNeuron= deviceProp.sharedMemPerBlock/sizeNeuron * deviceProp.multiProcessorCount;
    numNeuron = numNeuron - ( numNeuron % deviceProp.warpSize);
    printf("Caracteristica= %d banderas=%d sizeNeuron= %d numneuronas=%d \n",sizeVectorCharacteristic,sizeVectorFlags,sizeNeuron,numNeuron);

    return numNeuron;
    /*
     *calcule neuronas que puede tener un bloque porque pregunte
     *que es lo maximo que cabe en memoria compartida
     *sugerencia es mejor preguntar por los hilos y las dimensiones
     *y ver cuantas neuronas puedo correr en funcion de las dimensiones de hilos
     *releer capitulo
     */
}

void MainWindow::createInterfacesTeaching()
{
    formsTeaching = new FormTeaching*[numSenses];
    isInactivateSense    = new bool[numSenses];

    for (int i = 0; i < numSenses; i++)
    {
        formsTeaching[i] = new FormTeaching;
        isInactivateSense   [i] = false;
    }

    formsTeaching[SIGHT]->setSenseTeaching(SIGHT);
    formsTeaching[HEARING]->setSenseTeaching(HEARING);

    connect(formsTeaching[ SIGHT ],SIGNAL(signalGoodAnswer(senses)),this,SLOT(finishGoodAnswer(senses)));
    connect(formsTeaching[HEARING],SIGNAL(signalGoodAnswer(senses)),this,SLOT(finishGoodAnswer(senses)));
    connect(formsTeaching[SIGHT]->getPtrButtonTeach(),SIGNAL(clicked()),this,SLOT(learnSight()));
    connect(formsTeaching[HEARING]->getPtrButtonTeach(),SIGNAL(clicked()),this,SLOT(learnHearing()));
    connect(formsTeaching[SIGHT]->getPtrButtonGoodAnswer(),SIGNAL(clicked()),this,SLOT(resetSight()));
    connect(formsTeaching[HEARING]->getPtrButtonGoodAnswer(),SIGNAL(clicked()),this,SLOT(resetHearing()));
    connect(formsTeaching[SIGHT]->getPtrButtonState(),SIGNAL(clicked()),this,SLOT(paintStateSight()));
    connect(formsTeaching[HEARING]->getPtrButtonState(),SIGNAL(clicked()),this,SLOT(paintStateHearing()));

    ui->horizontalLayoutInterfaceEar->addWidget  (formsTeaching[HEARING]);
    ui->horizontalLayoutInterfaceSight->addWidget(formsTeaching[SIGHT]);
}

void MainWindow::setFormsCheck(bool state)
{
    for (int i = 0; i < numSenses; i++)
        isInactivateSense[i] = state;
}

void MainWindow::printAllVectorSenses()
{
    for (register int i = 0; i < this->numSenses; i++) {
        printf("Sentido %d \n",i);
        for(register int s = 0 ;s < 2336; s++)
        {
            printf("neurona %d \n",s);

            for(register int j=0; j < 32 ;j++)
            {
                printf("v[%d]=%d \n",j,neuralSenses[i].vectorNeuron[s*32+j]);
            }
            for(register int k=0; k < 9 ;k++)
            {
                printf("f[%d]=%d \n",k,neuralSenses[i].vectorFlags[s*9+k]);
            }
            for(register int l=0; l < 16 ;l++)
            {
                printf("VB[%d]=%d \n",l,neuralSenses[i].binaryCharacteristic[s*16+l]);
            }
        }

    }
}

void MainWindow::printVectorSenses(senses sense)
{
    for(register int s = 0 ;s < *(neuralSenses[sense].ptr)+1; s++)
    {
        printf("neurona %d \n",s);

        for(register int j=0; j < 32 ;j++)
        {
            printf("v[%d]=%d \n",j,neuralSenses[sense].vectorNeuron[s*32+j]);
        }
        for(register int k=0; k < 9 ;k++)
        {
            printf("f[%d]=%d \n",k,neuralSenses[sense].vectorFlags[s*9+k]);
        }
    }
}

void MainWindow::printIdsNeuronHit(senses sense)
{
    printSense(sense);
    printf("id hits se tienen %d hits\n",*interface[sense].hits);
    for(register int s = 0 ;s < *interface[sense].hits; s++)
        printf("neurona %d \n",interface[sense].id[s]);
}

void MainWindow::printSense(senses sense)
{
    if(sense== HEARING)
        printf("sentido OIDO \n");
    else
        printf("sentido VISTA \n");
}

void MainWindow::printCategory(senses sense)
{
    printSense(sense);
    printf("categorias se tienen %d categorias\n",*interface[sense].hits);
    for(register int s = 0 ;s < *interface[sense].hits; s++)
        printf("categorias %d \n",interface[sense].arrayCategory[s]);
}

template <class T>
void MainWindow::freeGenericPtr(T *ptr)
{
    if(ptr!= NULL)
        delete ptr;
}

bool MainWindow::analyticsNeuron(){
    if( interface[HEARING].hits == 0  || interface[SIGHT].hits == 0)      //  si hay hits en Oido y  Vista estan activa
        return false;

    unsigned char sightCategory;
    unsigned char tempEarC;
    sightCategory = interface[SIGHT].arrayCategory[0];
    if(ambiguity(sightCategory))                                                       //  Existe la ambiguedad

        tempEarC = checkInRelationNet(); //donde llamar esto?




    return true;
}

bool MainWindow::ambiguity(unsigned char sightCat){
    for(int i=0; i<*interface[SIGHT].hits; i++){                          //  Si hay distintas categorias en el
        if(interface[SIGHT].arrayCategory[i] =! sightCat)            //  vector de categorias de la vista entonces
            return true;
    }
}



void MainWindow::buildRelation(unsigned char){

    relationSenses->vectorSight[relationSenses->sizeRelationNet]     = interface[SIGHT].id[1];
    relationSenses->vectorEar[relationSenses->sizeRelationNet]       = interface[HEARING].id[1];
    relationSenses->vectorWeight[relationSenses->sizeRelationNet]   = 0   ;
    relationSenses->sizeRelationNet++;

}

void MainWindow::initializeRelation(int numNeuron){                        //  Reserva MEmoria para los vectores
    int tamVector;                                                         //  de la red de relaciones ( ESFERA CULTURAL)
    tamVector  = numNeuron * sizeof(unsigned char);                        //  Deberia ponerlos todos 0

    relationSenses->vectorEar    = new unsigned char [ tamVector];
    relationSenses->vectorSight  = new unsigned char [ tamVector];
    relationSenses->vectorWeight = new unsigned char [ tamVector];
    relationSenses->sizeRelationNet = 0;
}

unsigned char MainWindow::checkInRelationNet(){                                  //Chequea si existe una relacion
    unsigned char *tempCatSight;
    unsigned char *tempCatEar;
    int *idRelationNet;

    tempCatSight = new unsigned char[* interface[SIGHT].hits];
    tempCatEar = new unsigned char[* interface[SIGHT].hits];
    idRelationNet = new int [*interface[SIGHT].hits];

    for(int i = 0;i<*interface[SIGHT].hits;i++){
        tempCatSight[i] = NULL;
        tempCatEar[i] = NULL;
        idRelationNet[i] =-1;
    }

    bool ya = false;

    for(int j = 0; j<*interface[SIGHT].hits; j++){
        for(int i = 0; i < relationSenses->sizeRelationNet; i++){
            if(interface[SIGHT].arrayCategory[j] == relationSenses->vectorSight[i]){
                for(int k=0;k<j;k++){
                    if(tempCatSight[k] ==  relationSenses->vectorSight[i])
                        ya = true;
                    if(!ya){
                        tempCatSight[j] = relationSenses->vectorSight[i];
                        tempCatEar[j] = relationSenses->vectorEar[i];
                        idRelationNet[j] = i;
                        break;
                    }
                }
            }
        }
    }
    int idmax;
    int max = -1;
    for(int i = 0; i < *interface[SIGHT].hits; i++){
        if(idRelationNet[i]>=0)
            if(relationSenses->vectorWeight[idRelationNet[i]]>max){
                max = relationSenses->vectorWeight[idRelationNet[i]];
                idmax = idRelationNet[i];
            }
    }
    for(int i = 0; i < *interface[SIGHT].hits; i++){
        actualiceCategory(relationSenses->vectorSight[idmax],relationSenses->vectorEar[idRelationNet[i]]);
        interface[SIGHT].arrayCategory[i] = relationSenses->vectorSight[idmax];                 //en la interfaz tmb?
        //relationSenses->vectorSight[idRelationNet[i]] = relationSenses->vectorSight[idmax];
        relationSenses->vectorWeight[idmax]++;
    }
    //return relationSenses->vectorEar[i];
}




void MainWindow::actualiceCategory(unsigned char earCat, unsigned char sightid){
    neuralSenses[SIGHT].vectorFlags[sightid * SIZE_FLAGS + CAT ] = earCat;
}


/*==================================TESIS RICARDO GRATEROL====================================*/

void MainWindow::on_pushButtonBip_clicked() {}

void MainWindow::printCountNetwork() {
    for(register int i=0; i<kNeuron ; i++) {
        printf("Neurona: %d\n", countNetwork->vectorNetworkCount[i]);
    }
}

void MainWindow::startCount()
{
    ui->pushButton_startCount->hide();
    ui->textBrowser->show();
    ui->pushButton_stopCount->show();
    ui->pushButton_stopCount->show();
    ui->pushButton_teachBip->show();
    ui->pushButton_teachClack->show();
    ui->pushButton_teachClack->setEnabled(false);

}

void MainWindow::stopCount() {
    ui->pushButton_startCount->show();
    ui->textBrowser->hide();
    ui->pushButton_stopCount->hide();
    ui->pushButton_stopCount->hide();
    ui->pushButton_teachBip->hide();
    ui->pushButton_teachClack->hide();
}


void MainWindow::countProtocol()
{
    ui->pushButton_teachClack->setEnabled(true);
   /*if(kNeuron==0) {
        orderNetwork->countNet[orderNeuron].vectorNetworkCount[kNeuron]=1;
        orderNetwork->countNet[orderNeuron].vectorPointerCount[kNeuron]=kNeuron+1;
    }

    if(kNeuron!=0) {
        orderNetwork->countNet[orderNeuron].vectorNetworkCount[kNeuron]=1;
        orderNetwork->countNet[orderNeuron].vectorPointerCount[kNeuron]=kNeuron+1;
    }
    if(orderNetwork->countNet[orderNeuron].vectorNetworkCount[kNeuron]!=1) {
        orderNetwork->countNet[orderNeuron].vectorNetworkCount[kNeuron]=1;
        orderNetwork->countNet[orderNeuron].vectorPointerCount[kNeuron]=orderNeuron+1;
    }
    else {
        std::cout<<"Numero ya Conocido"<<endl;
    }
    //activateInterface(false);*/
    if(countNetwork->vectorNetworkCount[kNeuron]== 1) {
        std::cout<<"Numero ya Conocido"<<endl;
    }
    else {
        countNetwork->vectorNetworkCount[kNeuron]= 1;
        countNetwork->bipPointer[kNeuron]=kNeuron+1;
        std::cout<<"Numero Desconocido"<<endl;
    }
    QString cN = QString::number(kNeuron);
    QString oN = QString::number(orderNeuron);
    /*if(orderNetwork->countNet[kNeuron].vectorNetworkCount[kNeuron] == 1 ) {
        ui->textBrowser->setText("Numero ya conocido\nNeurona K: "+ cN+"\nNeurona O: "+oN);
    }
    else {
        ui->textBrowser->setText("Numero desconocido\nNeurona K: "+ cN+"\nNeurona O: "+oN);
    }*/
    kNeuron++;
}

void MainWindow::orderProtocol() {
    ui->pushButton_teachClack->setEnabled(false);
    /*orderNetwork->countNet[orderNeuron].vectorNetworkCount[kNeuron]=1;
    orderNetwork->countNet[orderNeuron].vectorPointerCount[kNeuron]=0;*/

   // orderNetwork->countNet[kNeuron].vectorNetworkCount[kNeuron] = 1;
   // orderNetwork->countNet[kNeuron].vectorPointerCount[kNeuron]= countNetwork->vectorPointerCount[kNeuron]-1;

    //orderNeuron++;

    kNeuron=1;
    orderNeuron=1;
    QString cN = QString::number(kNeuron);
    QString oN = QString::number(kNeuron);
    ui->textBrowser->setText("Numero Asimilado \nNeurona K: "+ cN+"\nNeurona O: "+oN);
}

/*QString getStringFromUnsignedChar(unsigned char *str)
{

    QString s;
    QString result = "";
    int rev = strlen(str);

    // Print String in Reverse order....
    for ( int i = 0; i<rev; i++)
        {
           s = QString("%1").arg(str[i],0,16);

           if(s == "0"){
              s="00";
             }
         result.append(s);

         }
   return result;
}*/
