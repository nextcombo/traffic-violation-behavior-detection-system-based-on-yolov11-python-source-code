# Traffic Violation Behavior Detection System Based on Yolov11 Python Source Code + PyTorch Model + Evaluation Indicator Curve + High-quality GUI Interface

[Algorithm Introduction]With the acceleration of urbanization, the surge in traffic volume has made traffic safety issues increasingly severe.

[Algorithm Introduction]

As urbanization accelerates, the surge in traffic volume has made traffic safety issues increasingly severe. Traffic violations not only frequently cause traffic accidents but also pose significant challenges to urban traffic management and public safety. To actively address this issue, the construction of intelligent transportation systems has become increasingly crucial. In recent years, with the rapid development of deep learning technology, especially the rise of the YOLO (You Only Look Once) series object detection algorithms, new paths have been opened for the automatic detection of traffic violation behaviors.

This study focuses on utilizing the YOLOv11 model to develop an efficient traffic violation detection system. We meticulously constructed a dataset containing 5056 images, covering various traffic violation categories such as pedestrians crossing the road illegally, pedestrians illegally entering zebra crossings, failure to wear helmets, normal driving, passenger violations, and traffic signal violations. This diverse dataset provides a solid foundation for model training, helping to enhance the accuracy and generalization capabilities of the detection.The dataset consists of seven main categories, covering various traffic violations, specifically: Crossing_Violation (Crossing Violation), Crosswalk_Violation (Pedestrian Violation), Helmet_Violation (Helmet Violation), Normal (Normal Behavior), Passenger_Violation (Passenger Violation), Pedestrian_Violation (Pedestrian Violation), and Trafficlight_Violation (Traffic Light Violation).

By calling the YOLOv11 model, we strive to enhance its detection speed while maintaining its efficiency. This system will provide strong real-time monitoring and data analysis support for traffic management departments, as well as send safety warnings to traffic participants, helping to enhance public awareness of traffic civility and safety. Furthermore, this deep learning-based traffic violation detection system lays a solid foundation for the future development of intelligent transportation systems, promoting comprehensive intelligentization in the field of transportation. Therefore, this study not only has profound theoretical value but also has broad practical application prospects.

[Effect Display]
![](01-e3026a05.jpg)

![](02-348e503b.jpg) 

[Training Dataset Introduction]

Dataset format: Pascal VOC format + YOLO format (txt files without split paths, only containing jpg images and corresponding VOC format xml files and yolo format txt files)
Number of images (number of jpg files): 5056
Number of annotations (number of xml files): 5056
Number of annotations (number of txt files): 5056
Number of annotation categories: 7
Annotation category names (note that the order in the yolo format does not correspond to this, but is based on the labels folder's classes.txt): ["Crossing_Violation", "Crosswalk_Violation", "Helmet_Violation", "Normal", "Passenger_Violation", "Pedestrian_Violation", "Trafficlight_Violation"]
Number of bounding boxes per category:
Crossing_Violation = 896
Crosswalk_Violation = 1056
Helmet_Violation = 486
Normal = 641
Passenger_Violation = 589
Pedestrian_Violation = 1389
Trafficlight_Violation = 278
Total bounding boxes: 5335
Using labelImg for annotation
Annotation rules: draw bounding boxes for each category
Important note: No guarantees are made regarding the accuracy or weights of the trained models or files. The dataset provides accurate and reasonable annotations only.

Image Preview:

![](03-687a0441.png)

Annotation example:

![](04-7bd38bf3.jpg)

[Training Information]

|

Parameters

 ||

Value

 |
|

Number of training set images

 ||

4669

 |
|

Number of images in the validation set

 ||

390

 |
|

Training map

 ||

97.8%

 |
|

Training Accuracy

 ||

97.2%

 |
|

Training recall rate

 ||

97.6%

 |

Evaluation accuracy information for the validation set:

|

Class

 ||

Images

 ||

Instances

 ||

P

 ||

R

 ||

mAP50

 ||

mAP50-95

 |
|

all

 ||

390

 ||

405

 ||

0.892

 ||

0.88

 ||

0.916

 ||

0.545

 |
|

Crossing_Violation

 ||

72

 ||

72

 ||

0.972

 ||

0.977

 ||

0.978

 ||

0.504

 |
|

Crosswalk_Violation

 ||

88

 ||

88

 ||

0.972

 ||

0.977

 ||

0.97

 ||

0.465

 |
|

Helmet_Violation

 ||

25

 ||

27

 ||

0.744

 ||

0.815

 ||

0.874

 ||

0.592

 |
|

Normal

 ||

40

 ||

48

 ||

0.856

 ||

0.792

 ||

0.849

 ||

0.49

 |
|

Passenger_Violation

 ||

47

 ||

47

 ||

0.965

 ||

0.979

 ||

0.993

 ||

0.867

 |
|

Pedestrian_Violation

 ||

98

 ||

102

 ||

0.847

 ||

0.812

 ||

0.883

 ||

0.442

 |
|

Trafficlight_Violation

 ||

21

 ||

21

 ||

0.889

 ||

0.81

 ||

0.868

 ||

0.452

 |

[Interface Code]

-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-

```
class Ui_MainWindow(QtWidgets.QMainWindow):    signal = QtCore.pyqtSignal(str, str)
    def setupUi(self):        self.setObjectName("MainWindow")        self.resize(1280, 728)        self.centralwidget = QtWidgets.QWidget(self)        self.centralwidget.setObjectName("centralwidget")
        self.weights_dir = './weights'
        self.picture = QtWidgets.QLabel(self.centralwidget)        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))        self.picture.setStyleSheet("background:black")        self.picture.setObjectName("picture")        self.picture.setScaledContents(True)        self.label_2 = QtWidgets.QLabel(self.centralwidget)        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))        self.label_2.setObjectName("label_2")        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))        self.cb_weights.setObjectName("cb_weights")        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))        self.label_3.setObjectName("label_3")        self.hs_conf = QtWidgets.QSlider(self.centralwidget)        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 181, 22))        self.hs_conf.setProperty("value", 25)        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)        self.hs_conf.setObjectName("hs_conf")        self.hs_conf.valueChanged.connect(self.conf_change)        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)        self.dsb_conf.setGeometry(QtCore.QRect(200, 100, 51, 22))        self.dsb_conf.setMaximum(1.0)        self.dsb_conf.setSingleStep(0.01)        self.dsb_conf.setProperty("value", 0.25)        self.dsb_conf.setObjectName("dsb_conf")        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)        self.dsb_iou.setGeometry(QtCore.QRect(200, 160, 51, 22))        self.dsb_iou.setMaximum(1.0)        self.dsb_iou.setSingleStep(0.01)        self.dsb_iou.setProperty("value", 0.45)        self.dsb_iou.setObjectName("dsb_iou")        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)        self.hs_iou = QtWidgets.QSlider(self.centralwidget)        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 181, 22))        self.hs_iou.setProperty("value", 45)        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)        self.hs_iou.setObjectName("hs_iou")        self.hs_iou.valueChanged.connect(self.iou_change)        self.label_4 = QtWidgets.QLabel(self.centralwidget)        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))        self.label_4.setObjectName("label_4")        self.label_5 = QtWidgets.QLabel(self.centralwidget)        self.label_5.setGeometry(QtCore.QRect(10, 210, 72, 21))        self.label_5.setObjectName("label_5")        self.le_res = QtWidgets.QTextEdit(self.centralwidget)        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 400))        self.le_res.setObjectName("le_res")        self.setCentralWidget(self.centralwidget)        self.menubar = QtWidgets.QMenuBar(self)        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))        self.menubar.setObjectName("menubar")        self.setMenuBar(self.menubar)        self.statusbar = QtWidgets.QStatusBar(self)        self.statusbar.setObjectName("statusbar")        self.setStatusBar(self.statusbar)        self.toolBar = QtWidgets.QToolBar(self)        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)        self.toolBar.setObjectName("toolBar")        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)        self.actionopenpic = QtWidgets.QAction(self)        icon = QtGui.QIcon()        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.actionopenpic.setIcon(icon)        self.actionopenpic.setObjectName("actionopenpic")        self.actionopenpic.triggered.connect(self.open_image)        self.action = QtWidgets.QAction(self)        icon1 = QtGui.QIcon()        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.action.setIcon(icon1)        self.action.setObjectName("action")        self.action.triggered.connect(self.open_video)        self.action_2 = QtWidgets.QAction(self)        icon2 = QtGui.QIcon()        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.action_2.setIcon(icon2)        self.action_2.setObjectName("action_2")        self.action_2.triggered.connect(self.open_camera)
        self.actionexit = QtWidgets.QAction(self)        icon3 = QtGui.QIcon()        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)        self.actionexit.setIcon(icon3)        self.actionexit.setObjectName("actionexit")        self.actionexit.triggered.connect(self.exit)
        self.toolBar.addAction(self.actionopenpic)        self.toolBar.addAction(self.action)        self.toolBar.addAction(self.action_2)        self.toolBar.addAction(self.actionexit)
        self.retranslateUi()        QtCore.QMetaObject.connectSlotsByName(self)        self.init_all()
```

[Common Evaluation Parameters Introduction]

In the task of object detection, evaluating the performance of a model is crucial. The terms you mentioned are commonly used metrics for assessing the performance of a model. Here are detailed explanations of these terms:

1. **Class**

: 

-

This usually refers to the categories of objects that the model is designed to detect. For example, a model may be trained to detect different categories of objects such as vehicles, pedestrians, or animals.

1. **Images**

: 

-

Represents the number of images in the validation set. The validation set is a dataset used to evaluate model performance, separate from the training set to ensure fairness in evaluation results.

1. **Instances**

: 

-

The total number of target objects in all images. This includes the sum of all category objects, for example, if the validation set contains 100 images, each image has an average of 5 target objects, then Instances is 500.

1. **P (Precision)**

: 

-

Precision is the proportion of actual positive samples predicted by the model. The formula is: Precision = TP / (TP + FP), where TP represents true positives (True Positives) and FP represents false positives (False Positives).

1. **R (Recall)**

: 

-

Recall is the proportion of all true positives correctly predicted by the model. The formula is: Recall = TP / (TP + FN), where FN represents false negatives.

1. **mAP50**

: 

-

Mean Average Precision (mAP) at an IoU threshold of 0.5, where IoU is the metric used to measure the degree of overlap between predicted bounding boxes and ground truth bounding boxes. mAP is a comprehensive metric that considers both precision and recall when evaluating model performance across different recall levels. At an IoU of 0.5, if the overlap between the predicted bounding box and the ground truth bounding box reaches or exceeds 50%, it is considered correct.

1. **mAP50-95**

: 

-

This represents the average accuracy of the model across IoU values ranging from 0.5 to 0.95 (with a step of 0.05). This is a more stringent evaluation criterion, demanding higher overlap between predicted bounding boxes and ground truth boxes. In object detection tasks, higher IoU thresholds mean that models need to be more accurate in their target object localization. The computation of mAP50-95 takes into account multiple IoU thresholds ranging from lenient to strict, thus providing a more comprehensive assessment of the model's performance.

These indicators collectively form an important framework for evaluating the performance of target detection models. By comparing the performance of different models on these indicators, It can be determined which model may be more effective in practical applications.

[Usage Steps]

Usage Steps:
1. Install the YOLOv11 environment according to the official framework ultralytics installation tutorial, and install PyQt5.
2. After switching to the YOLOv11 environment you have installed, and switch to the source code directory, execute python main.py to start the interface, and perform the corresponding operations.

[Provided Files]

Python source code
YOLOv11n.pt model
Training map, P, R curves (in weights\results.png)
600 test images (under test_img folder)

Note: The source code provides the training dataset.


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
