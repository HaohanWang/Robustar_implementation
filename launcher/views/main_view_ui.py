# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'launcher_v2.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_RobustarLauncher(object):
    def setupUi(self, RobustarLauncher):
        if not RobustarLauncher.objectName():
            RobustarLauncher.setObjectName(u"RobustarLauncher")
        RobustarLauncher.resize(1000, 850)
        font = QFont()
        font.setFamily(u"Arial")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        RobustarLauncher.setFont(font)
        icon = QIcon()
        icon.addFile(u"../doc/logo_short.png", QSize(), QIcon.Normal, QIcon.Off)
        RobustarLauncher.setWindowIcon(icon)
        RobustarLauncher.setStyleSheet(u"QWidget{font: 8pt \"Arial\"}\n"
"QPushButton{background-color: #E1E1E1; \n"
"			font: 8pt Arial;\n"
"			border: 1px solid #ADADAD;}\n"
"QPushButton:hover {\n"
"   background-color: #E3ECF3;\n"
"   border: 1px solid #3287CA;\n"
"}")
        self.layoutWidget = QWidget(RobustarLauncher)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(0, 0, 1001, 841))
        self.verticalLayout = QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.headerLayout = QHBoxLayout()
        self.headerLayout.setObjectName(u"headerLayout")
        self.header = QLabel(self.layoutWidget)
        self.header.setObjectName(u"header")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.header.sizePolicy().hasHeightForWidth())
        self.header.setSizePolicy(sizePolicy)
        self.header.setMaximumSize(QSize(164, 42))
        self.header.setTextFormat(Qt.AutoText)
        self.header.setPixmap(QPixmap(u"../doc/logo_long.png"))
        self.header.setScaledContents(True)
        self.header.setIndent(-1)

        self.headerLayout.addWidget(self.header)


        self.verticalLayout.addLayout(self.headerLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(15, -1, 11, -1)
        self.buttonFrame = QFrame(self.layoutWidget)
        self.buttonFrame.setObjectName(u"buttonFrame")
        self.buttonFrame.setFont(font)
        self.buttonFrame.setStyleSheet(u"#buttonFrame{background: #F9F9F9}\n"
"")
        self.buttonFrame.setFrameShape(QFrame.StyledPanel)
        self.buttonFrame.setFrameShadow(QFrame.Raised)
        self.layoutWidget1 = QWidget(self.buttonFrame)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(0, 0, 161, 771))
        self.buttonLayout = QVBoxLayout(self.layoutWidget1)
        self.buttonLayout.setSpacing(60)
        self.buttonLayout.setObjectName(u"buttonLayout")
        self.buttonLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.buttonLayout.setContentsMargins(0, 60, 0, 0)
        self.loadButtonLayout = QHBoxLayout()
        self.loadButtonLayout.setObjectName(u"loadButtonLayout")
        self.loadButtonLayout.setContentsMargins(-1, 0, -1, -1)
        self.horizontalSpacer_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.loadButtonLayout.addItem(self.horizontalSpacer_9)

        self.loadProfileButton = QPushButton(self.layoutWidget1)
        self.loadProfileButton.setObjectName(u"loadProfileButton")
        sizePolicy1 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.loadProfileButton.sizePolicy().hasHeightForWidth())
        self.loadProfileButton.setSizePolicy(sizePolicy1)
        self.loadProfileButton.setMinimumSize(QSize(100, 25))
        self.loadProfileButton.setMaximumSize(QSize(160, 16777215))
        self.loadProfileButton.setFont(font)
        self.loadProfileButton.setStyleSheet(u"")

        self.loadButtonLayout.addWidget(self.loadProfileButton)

        self.horizontalSpacer_10 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.loadButtonLayout.addItem(self.horizontalSpacer_10)


        self.buttonLayout.addLayout(self.loadButtonLayout)

        self.saveButtonLayout = QHBoxLayout()
        self.saveButtonLayout.setObjectName(u"saveButtonLayout")
        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.saveButtonLayout.addItem(self.horizontalSpacer_7)

        self.saveProfileButton = QPushButton(self.layoutWidget1)
        self.saveProfileButton.setObjectName(u"saveProfileButton")
        sizePolicy1.setHeightForWidth(self.saveProfileButton.sizePolicy().hasHeightForWidth())
        self.saveProfileButton.setSizePolicy(sizePolicy1)
        self.saveProfileButton.setMinimumSize(QSize(100, 25))
        self.saveProfileButton.setMaximumSize(QSize(160, 16777215))
        self.saveProfileButton.setFont(font)

        self.saveButtonLayout.addWidget(self.saveProfileButton)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.saveButtonLayout.addItem(self.horizontalSpacer_8)


        self.buttonLayout.addLayout(self.saveButtonLayout)

        self.startButtonLayout = QHBoxLayout()
        self.startButtonLayout.setObjectName(u"startButtonLayout")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.startButtonLayout.addItem(self.horizontalSpacer_5)

        self.startServerButton = QPushButton(self.layoutWidget1)
        self.startServerButton.setObjectName(u"startServerButton")
        sizePolicy1.setHeightForWidth(self.startServerButton.sizePolicy().hasHeightForWidth())
        self.startServerButton.setSizePolicy(sizePolicy1)
        self.startServerButton.setMinimumSize(QSize(100, 25))
        self.startServerButton.setMaximumSize(QSize(160, 16777215))
        self.startServerButton.setFont(font)
        self.startServerButton.setLayoutDirection(Qt.LeftToRight)

        self.startButtonLayout.addWidget(self.startServerButton)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.startButtonLayout.addItem(self.horizontalSpacer_6)


        self.buttonLayout.addLayout(self.startButtonLayout)

        self.stopButtoLayout = QHBoxLayout()
        self.stopButtoLayout.setObjectName(u"stopButtoLayout")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.stopButtoLayout.addItem(self.horizontalSpacer_3)

        self.stopServerButton = QPushButton(self.layoutWidget1)
        self.stopServerButton.setObjectName(u"stopServerButton")
        sizePolicy1.setHeightForWidth(self.stopServerButton.sizePolicy().hasHeightForWidth())
        self.stopServerButton.setSizePolicy(sizePolicy1)
        self.stopServerButton.setMinimumSize(QSize(100, 25))
        self.stopServerButton.setMaximumSize(QSize(160, 16777215))
        self.stopServerButton.setFont(font)

        self.stopButtoLayout.addWidget(self.stopServerButton)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.stopButtoLayout.addItem(self.horizontalSpacer_4)


        self.buttonLayout.addLayout(self.stopButtoLayout)

        self.deleteButtonLayout = QHBoxLayout()
        self.deleteButtonLayout.setObjectName(u"deleteButtonLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.deleteButtonLayout.addItem(self.horizontalSpacer)

        self.deleteServerButton = QPushButton(self.layoutWidget1)
        self.deleteServerButton.setObjectName(u"deleteServerButton")
        sizePolicy1.setHeightForWidth(self.deleteServerButton.sizePolicy().hasHeightForWidth())
        self.deleteServerButton.setSizePolicy(sizePolicy1)
        self.deleteServerButton.setMinimumSize(QSize(100, 25))
        self.deleteServerButton.setMaximumSize(QSize(160, 16777215))
        self.deleteServerButton.setFont(font)

        self.deleteButtonLayout.addWidget(self.deleteServerButton)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.deleteButtonLayout.addItem(self.horizontalSpacer_2)


        self.buttonLayout.addLayout(self.deleteButtonLayout)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        self.buttonLayout.addItem(self.verticalSpacer_2)

        self.buttonLayout.setStretch(0, 1)
        self.buttonLayout.setStretch(1, 1)
        self.buttonLayout.setStretch(2, 1)
        self.buttonLayout.setStretch(3, 1)
        self.buttonLayout.setStretch(4, 1)
        self.buttonLayout.setStretch(5, 2)

        self.horizontalLayout_2.addWidget(self.buttonFrame)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(-1, 0, -1, 0)
        self.tabWidget = QTabWidget(self.layoutWidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setFont(font)
        self.createTab = QWidget()
        self.createTab.setObjectName(u"createTab")
        self.horizontalLayoutWidget_7 = QWidget(self.createTab)
        self.horizontalLayoutWidget_7.setObjectName(u"horizontalLayoutWidget_7")
        self.horizontalLayoutWidget_7.setGeometry(QRect(20, 10, 771, 411))
        self.horizontalLayout_6 = QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.dockerGroupBox = QGroupBox(self.horizontalLayoutWidget_7)
        self.dockerGroupBox.setObjectName(u"dockerGroupBox")
        self.dockerGroupBox.setFont(font)
        self.layoutWidget2 = QWidget(self.dockerGroupBox)
        self.layoutWidget2.setObjectName(u"layoutWidget2")
        self.layoutWidget2.setGeometry(QRect(10, 20, 361, 131))
        self.verticalLayout_6 = QVBoxLayout(self.layoutWidget2)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_8 = QLabel(self.layoutWidget2)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)

        self.horizontalLayout_8.addWidget(self.label_8)

        self.nameInput = QLineEdit(self.layoutWidget2)
        self.nameInput.setObjectName(u"nameInput")
        sizePolicy1.setHeightForWidth(self.nameInput.sizePolicy().hasHeightForWidth())
        self.nameInput.setSizePolicy(sizePolicy1)
        self.nameInput.setMinimumSize(QSize(0, 20))
        self.nameInput.setMaximumSize(QSize(175, 16777215))
        self.nameInput.setFont(font)
        self.nameInput.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.nameInput.setClearButtonEnabled(True)

        self.horizontalLayout_8.addWidget(self.nameInput)


        self.verticalLayout_6.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_3 = QLabel(self.layoutWidget2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)

        self.horizontalLayout_5.addWidget(self.label_3)

        self.versionComboBox = QComboBox(self.layoutWidget2)
        self.versionComboBox.addItem("")
        self.versionComboBox.addItem("")
        self.versionComboBox.setObjectName(u"versionComboBox")
        sizePolicy1.setHeightForWidth(self.versionComboBox.sizePolicy().hasHeightForWidth())
        self.versionComboBox.setSizePolicy(sizePolicy1)
        self.versionComboBox.setMaximumSize(QSize(175, 16777215))
        self.versionComboBox.setFont(font)

        self.horizontalLayout_5.addWidget(self.versionComboBox)


        self.verticalLayout_6.addLayout(self.horizontalLayout_5)


        self.verticalLayout_3.addWidget(self.dockerGroupBox)

        self.portGroupBox = QGroupBox(self.horizontalLayoutWidget_7)
        self.portGroupBox.setObjectName(u"portGroupBox")
        self.portGroupBox.setFont(font)
        self.layoutWidget3 = QWidget(self.portGroupBox)
        self.layoutWidget3.setObjectName(u"layoutWidget3")
        self.layoutWidget3.setGeometry(QRect(10, 20, 361, 211))
        self.verticalLayout_5 = QVBoxLayout(self.layoutWidget3)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_7 = QLabel(self.layoutWidget3)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font)

        self.horizontalLayout_9.addWidget(self.label_7)

        self.websitePortInput = QLineEdit(self.layoutWidget3)
        self.websitePortInput.setObjectName(u"websitePortInput")
        sizePolicy1.setHeightForWidth(self.websitePortInput.sizePolicy().hasHeightForWidth())
        self.websitePortInput.setSizePolicy(sizePolicy1)
        self.websitePortInput.setMaximumSize(QSize(175, 16777215))
        self.websitePortInput.setFont(font)
        self.websitePortInput.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.websitePortInput.setClearButtonEnabled(True)

        self.horizontalLayout_9.addWidget(self.websitePortInput)


        self.verticalLayout_5.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_11 = QLabel(self.layoutWidget3)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font)

        self.horizontalLayout_12.addWidget(self.label_11)

        self.backendPortInput = QLineEdit(self.layoutWidget3)
        self.backendPortInput.setObjectName(u"backendPortInput")
        sizePolicy1.setHeightForWidth(self.backendPortInput.sizePolicy().hasHeightForWidth())
        self.backendPortInput.setSizePolicy(sizePolicy1)
        self.backendPortInput.setMaximumSize(QSize(175, 16777215))
        self.backendPortInput.setFont(font)
        self.backendPortInput.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.backendPortInput.setClearButtonEnabled(True)

        self.horizontalLayout_12.addWidget(self.backendPortInput)


        self.verticalLayout_5.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_12 = QLabel(self.layoutWidget3)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setFont(font)

        self.horizontalLayout_13.addWidget(self.label_12)

        self.tensorboardPortInput = QLineEdit(self.layoutWidget3)
        self.tensorboardPortInput.setObjectName(u"tensorboardPortInput")
        sizePolicy1.setHeightForWidth(self.tensorboardPortInput.sizePolicy().hasHeightForWidth())
        self.tensorboardPortInput.setSizePolicy(sizePolicy1)
        self.tensorboardPortInput.setMaximumSize(QSize(175, 16777215))
        self.tensorboardPortInput.setFont(font)
        self.tensorboardPortInput.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.tensorboardPortInput.setClearButtonEnabled(True)

        self.horizontalLayout_13.addWidget(self.tensorboardPortInput)


        self.verticalLayout_5.addLayout(self.horizontalLayout_13)


        self.verticalLayout_3.addWidget(self.portGroupBox)

        self.verticalLayout_3.setStretch(0, 2)
        self.verticalLayout_3.setStretch(1, 3)

        self.horizontalLayout_3.addLayout(self.verticalLayout_3)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.pathGroupBox = QGroupBox(self.horizontalLayoutWidget_7)
        self.pathGroupBox.setObjectName(u"pathGroupBox")
        self.pathGroupBox.setFont(font)
        self.layoutWidget4 = QWidget(self.pathGroupBox)
        self.layoutWidget4.setObjectName(u"layoutWidget4")
        self.layoutWidget4.setGeometry(QRect(10, 20, 361, 371))
        self.verticalLayout_7 = QVBoxLayout(self.layoutWidget4)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setContentsMargins(0, -1, -1, -1)
        self.label_6 = QLabel(self.layoutWidget4)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMaximumSize(QSize(140, 16777215))
        self.label_6.setFont(font)

        self.horizontalLayout_16.addWidget(self.label_6)

        self.trainPathDisplay = QLineEdit(self.layoutWidget4)
        self.trainPathDisplay.setObjectName(u"trainPathDisplay")
        sizePolicy1.setHeightForWidth(self.trainPathDisplay.sizePolicy().hasHeightForWidth())
        self.trainPathDisplay.setSizePolicy(sizePolicy1)
        self.trainPathDisplay.setMinimumSize(QSize(175, 0))
        self.trainPathDisplay.setMaximumSize(QSize(104, 16777215))
        self.trainPathDisplay.setFont(font)
        self.trainPathDisplay.setMouseTracking(True)
        self.trainPathDisplay.setReadOnly(False)

        self.horizontalLayout_16.addWidget(self.trainPathDisplay)

        self.trainPathButton = QPushButton(self.layoutWidget4)
        self.trainPathButton.setObjectName(u"trainPathButton")
        sizePolicy1.setHeightForWidth(self.trainPathButton.sizePolicy().hasHeightForWidth())
        self.trainPathButton.setSizePolicy(sizePolicy1)
        self.trainPathButton.setMinimumSize(QSize(28, 0))
        self.trainPathButton.setMaximumSize(QSize(28, 16777215))
        self.trainPathButton.setFont(font)

        self.horizontalLayout_16.addWidget(self.trainPathButton)


        self.verticalLayout_7.addLayout(self.horizontalLayout_16)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.label_13 = QLabel(self.layoutWidget4)
        self.label_13.setObjectName(u"label_13")
        self.label_13.setMaximumSize(QSize(140, 16777215))
        self.label_13.setFont(font)

        self.horizontalLayout_17.addWidget(self.label_13)

        self.testPathDisplay = QLineEdit(self.layoutWidget4)
        self.testPathDisplay.setObjectName(u"testPathDisplay")
        self.testPathDisplay.setMinimumSize(QSize(175, 0))
        self.testPathDisplay.setMaximumSize(QSize(175, 16777215))
        self.testPathDisplay.setFont(font)

        self.horizontalLayout_17.addWidget(self.testPathDisplay)

        self.testPathButton = QPushButton(self.layoutWidget4)
        self.testPathButton.setObjectName(u"testPathButton")
        sizePolicy1.setHeightForWidth(self.testPathButton.sizePolicy().hasHeightForWidth())
        self.testPathButton.setSizePolicy(sizePolicy1)
        self.testPathButton.setMinimumSize(QSize(28, 0))
        self.testPathButton.setMaximumSize(QSize(28, 16777215))
        self.testPathButton.setFont(font)

        self.horizontalLayout_17.addWidget(self.testPathButton)


        self.verticalLayout_7.addLayout(self.horizontalLayout_17)

        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.label_14 = QLabel(self.layoutWidget4)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setMaximumSize(QSize(140, 16777215))
        self.label_14.setFont(font)

        self.horizontalLayout_18.addWidget(self.label_14)

        self.checkPointPathDisplay = QLineEdit(self.layoutWidget4)
        self.checkPointPathDisplay.setObjectName(u"checkPointPathDisplay")
        self.checkPointPathDisplay.setMaximumSize(QSize(175, 16777215))
        self.checkPointPathDisplay.setFont(font)

        self.horizontalLayout_18.addWidget(self.checkPointPathDisplay)

        self.checkPointPathButton = QPushButton(self.layoutWidget4)
        self.checkPointPathButton.setObjectName(u"checkPointPathButton")
        sizePolicy1.setHeightForWidth(self.checkPointPathButton.sizePolicy().hasHeightForWidth())
        self.checkPointPathButton.setSizePolicy(sizePolicy1)
        self.checkPointPathButton.setMinimumSize(QSize(28, 0))
        self.checkPointPathButton.setMaximumSize(QSize(28, 16777215))
        self.checkPointPathButton.setFont(font)

        self.horizontalLayout_18.addWidget(self.checkPointPathButton)


        self.verticalLayout_7.addLayout(self.horizontalLayout_18)

        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_15 = QLabel(self.layoutWidget4)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setMaximumSize(QSize(140, 16777215))
        self.label_15.setFont(font)

        self.horizontalLayout_19.addWidget(self.label_15)

        self.influencePathDisplay = QLineEdit(self.layoutWidget4)
        self.influencePathDisplay.setObjectName(u"influencePathDisplay")
        self.influencePathDisplay.setMaximumSize(QSize(175, 16777215))
        self.influencePathDisplay.setFont(font)

        self.horizontalLayout_19.addWidget(self.influencePathDisplay)

        self.influencePathButton = QPushButton(self.layoutWidget4)
        self.influencePathButton.setObjectName(u"influencePathButton")
        sizePolicy1.setHeightForWidth(self.influencePathButton.sizePolicy().hasHeightForWidth())
        self.influencePathButton.setSizePolicy(sizePolicy1)
        self.influencePathButton.setMinimumSize(QSize(28, 0))
        self.influencePathButton.setMaximumSize(QSize(28, 16777215))
        self.influencePathButton.setFont(font)

        self.horizontalLayout_19.addWidget(self.influencePathButton)


        self.verticalLayout_7.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_21 = QHBoxLayout()
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.label_9 = QLabel(self.layoutWidget4)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setMaximumSize(QSize(140, 16777215))
        self.label_9.setFont(font)

        self.horizontalLayout_21.addWidget(self.label_9)

        self.configFileDisplay = QLineEdit(self.layoutWidget4)
        self.configFileDisplay.setObjectName(u"configFileDisplay")
        sizePolicy1.setHeightForWidth(self.configFileDisplay.sizePolicy().hasHeightForWidth())
        self.configFileDisplay.setSizePolicy(sizePolicy1)
        self.configFileDisplay.setMinimumSize(QSize(104, 0))
        self.configFileDisplay.setMaximumSize(QSize(175, 16777215))
        self.configFileDisplay.setFont(font)
        self.configFileDisplay.setMouseTracking(True)
        self.configFileDisplay.setReadOnly(False)

        self.horizontalLayout_21.addWidget(self.configFileDisplay)

        self.configFileButton = QPushButton(self.layoutWidget4)
        self.configFileButton.setObjectName(u"configFileButton")
        sizePolicy1.setHeightForWidth(self.configFileButton.sizePolicy().hasHeightForWidth())
        self.configFileButton.setSizePolicy(sizePolicy1)
        self.configFileButton.setMinimumSize(QSize(28, 0))
        self.configFileButton.setMaximumSize(QSize(28, 16777215))
        self.configFileButton.setFont(font)

        self.horizontalLayout_21.addWidget(self.configFileButton)


        self.verticalLayout_7.addLayout(self.horizontalLayout_21)


        self.verticalLayout_4.addWidget(self.pathGroupBox)

        self.verticalLayout_4.setStretch(0, 4)

        self.horizontalLayout_3.addLayout(self.verticalLayout_4)


        self.horizontalLayout_6.addLayout(self.horizontalLayout_3)

        self.tabWidget.addTab(self.createTab, "")
        self.manageTab = QWidget()
        self.manageTab.setObjectName(u"manageTab")
        self.layoutWidget5 = QWidget(self.manageTab)
        self.layoutWidget5.setObjectName(u"layoutWidget5")
        self.layoutWidget5.setGeometry(QRect(20, 10, 771, 401))
        self.horizontalLayout_20 = QHBoxLayout(self.layoutWidget5)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.horizontalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.groupBox_2 = QGroupBox(self.layoutWidget5)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.exitedListWidget = QListWidget(self.groupBox_2)
        self.exitedListWidget.setObjectName(u"exitedListWidget")
        self.exitedListWidget.setGeometry(QRect(10, 30, 361, 161))

        self.verticalLayout_8.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.layoutWidget5)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setFont(font)
        self.createdListWidget = QListWidget(self.groupBox_3)
        self.createdListWidget.setObjectName(u"createdListWidget")
        self.createdListWidget.setGeometry(QRect(10, 20, 361, 171))

        self.verticalLayout_8.addWidget(self.groupBox_3)

        self.verticalLayout_8.setStretch(0, 1)
        self.verticalLayout_8.setStretch(1, 1)

        self.horizontalLayout_20.addLayout(self.verticalLayout_8)

        self.groupBox = QGroupBox(self.layoutWidget5)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setFont(font)
        self.runningListWidget = QListWidget(self.groupBox)
        self.runningListWidget.setObjectName(u"runningListWidget")
        self.runningListWidget.setGeometry(QRect(10, 30, 361, 361))

        self.horizontalLayout_20.addWidget(self.groupBox)

        self.refreshListWidgetsButton = QPushButton(self.manageTab)
        self.refreshListWidgetsButton.setObjectName(u"refreshListWidgetsButton")
        self.refreshListWidgetsButton.setGeometry(QRect(717, 416, 75, 25))
        self.tabWidget.addTab(self.manageTab, "")

        self.verticalLayout_2.addWidget(self.tabWidget)

        self.tabWidget_2 = QTabWidget(self.layoutWidget)
        self.tabWidget_2.setObjectName(u"tabWidget_2")
        self.tabWidget_2.setFont(font)
        self.tabWidget_2.setStyleSheet(u"")
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.promptBrowser = QTextBrowser(self.tab)
        self.promptBrowser.setObjectName(u"promptBrowser")
        self.promptBrowser.setGeometry(QRect(-1, -1, 821, 301))
        self.promptBrowser.setStyleSheet(u"")
        self.tabWidget_2.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.detailBrowser = QTextBrowser(self.tab_2)
        self.detailBrowser.setObjectName(u"detailBrowser")
        self.detailBrowser.setGeometry(QRect(-1, -1, 821, 291))
        self.tabWidget_2.addTab(self.tab_2, "")
        self.tab_3 = QWidget()
        self.tab_3.setObjectName(u"tab_3")
        self.logBrowser = QTextBrowser(self.tab_3)
        self.logBrowser.setObjectName(u"logBrowser")
        self.logBrowser.setGeometry(QRect(-1, -1, 821, 291))
        self.tabWidget_2.addTab(self.tab_3, "")

        self.verticalLayout_2.addWidget(self.tabWidget_2)

        self.verticalLayout_2.setStretch(0, 3)
        self.verticalLayout_2.setStretch(1, 2)

        self.horizontalLayout_2.addLayout(self.verticalLayout_2)

        self.horizontalLayout_2.setStretch(0, 2)
        self.horizontalLayout_2.setStretch(1, 10)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 11)

        self.retranslateUi(RobustarLauncher)

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(RobustarLauncher)
    # setupUi

    def retranslateUi(self, RobustarLauncher):
        RobustarLauncher.setWindowTitle(QCoreApplication.translate("RobustarLauncher", u"Robustar Launcher", None))
        self.header.setText("")
        self.loadProfileButton.setText(QCoreApplication.translate("RobustarLauncher", u"Load Profile", None))
        self.saveProfileButton.setText(QCoreApplication.translate("RobustarLauncher", u"Save Profile", None))
        self.startServerButton.setText(QCoreApplication.translate("RobustarLauncher", u"Start Server", None))
        self.stopServerButton.setText(QCoreApplication.translate("RobustarLauncher", u"Stop Server", None))
        self.deleteServerButton.setText(QCoreApplication.translate("RobustarLauncher", u"Delete Server", None))
        self.dockerGroupBox.setTitle(QCoreApplication.translate("RobustarLauncher", u"Docker", None))
        self.label_8.setText(QCoreApplication.translate("RobustarLauncher", u"Docker Container Name", None))
        self.nameInput.setText(QCoreApplication.translate("RobustarLauncher", u"robustar", None))
        self.label_3.setText(QCoreApplication.translate("RobustarLauncher", u"Docker Image Version", None))
        self.versionComboBox.setItemText(0, QCoreApplication.translate("RobustarLauncher", u"cuda11.1-0.0.1-beta", None))
        self.versionComboBox.setItemText(1, QCoreApplication.translate("RobustarLauncher", u"cpu-0.0.1-beta", None))

        self.versionComboBox.setCurrentText(QCoreApplication.translate("RobustarLauncher", u"cuda11.1-0.0.1-beta", None))
        self.versionComboBox.setPlaceholderText("")
        self.portGroupBox.setTitle(QCoreApplication.translate("RobustarLauncher", u"Port", None))
        self.label_7.setText(QCoreApplication.translate("RobustarLauncher", u"Website Port", None))
        self.websitePortInput.setText(QCoreApplication.translate("RobustarLauncher", u"8000", None))
        self.label_11.setText(QCoreApplication.translate("RobustarLauncher", u"Backend Port", None))
        self.backendPortInput.setText(QCoreApplication.translate("RobustarLauncher", u"6848", None))
        self.label_12.setText(QCoreApplication.translate("RobustarLauncher", u"Tensorboard Port", None))
        self.tensorboardPortInput.setText(QCoreApplication.translate("RobustarLauncher", u"6006", None))
        self.pathGroupBox.setTitle(QCoreApplication.translate("RobustarLauncher", u"Path", None))
        self.label_6.setText(QCoreApplication.translate("RobustarLauncher", u"Train Set Path", None))
        self.trainPathDisplay.setText("")
        self.trainPathButton.setText(QCoreApplication.translate("RobustarLauncher", u"...", None))
        self.label_13.setText(QCoreApplication.translate("RobustarLauncher", u"Test Set Path", None))
        self.testPathDisplay.setText("")
        self.testPathButton.setText(QCoreApplication.translate("RobustarLauncher", u"...", None))
        self.label_14.setText(QCoreApplication.translate("RobustarLauncher", u"Check Point Path", None))
        self.checkPointPathDisplay.setText("")
        self.checkPointPathButton.setText(QCoreApplication.translate("RobustarLauncher", u"...", None))
        self.label_15.setText(QCoreApplication.translate("RobustarLauncher", u"Influence Result Path", None))
        self.influencePathDisplay.setText("")
        self.influencePathButton.setText(QCoreApplication.translate("RobustarLauncher", u"...", None))
        self.label_9.setText(QCoreApplication.translate("RobustarLauncher", u"Config File Path", None))
        self.configFileDisplay.setText("")
        self.configFileButton.setText(QCoreApplication.translate("RobustarLauncher", u"...", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.createTab), QCoreApplication.translate("RobustarLauncher", u"Create", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("RobustarLauncher", u"Exited", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("RobustarLauncher", u"Created", None))
        self.groupBox.setTitle(QCoreApplication.translate("RobustarLauncher", u"Running", None))
        self.refreshListWidgetsButton.setText(QCoreApplication.translate("RobustarLauncher", u"Refresh", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.manageTab), QCoreApplication.translate("RobustarLauncher", u"Manage", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab), QCoreApplication.translate("RobustarLauncher", u"Prompts", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_2), QCoreApplication.translate("RobustarLauncher", u"Details", None))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), QCoreApplication.translate("RobustarLauncher", u"Logs", None))
    # retranslateUi
