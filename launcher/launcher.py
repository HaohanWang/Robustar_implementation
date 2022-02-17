import json
import time

import docker

from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QFileDialog, QWidget, QListWidget, QListWidgetItem
from PySide2.QtCore import Signal, QObject, Qt
from threading import Thread

class CustomSignals(QObject):
    # Custom signal for printing message in messageBrowser
    printMessageSignal = Signal(str)

    # Custom signal for adding items in running or exited container list widget
    addItemSignal = Signal(QListWidget, str)

    # Custom signal for removing item from running or exited container list widget
    removeItemSignal = Signal(QListWidget, str)


class Launcher(QWidget):
    def __init__(self):
        # Initialize the UI
        super(Launcher, self).__init__()
        self.ui = QUiLoader().load('launcher_v2.ui')

        # Initialize the signal source object
        self.customSignals = CustomSignals()

        # Initialize the client to communicate with the Docker daemon
        self.client = docker.from_env()

        # Set the default path of the path choosing window
        self.cwd = '/'

        # Set the starting state of the server
        self.runningState = False

        # Set the default configuration
        self.configs = {
                        'containerName': 'robustar',
                        'imageVersion': 'cuda11.1-0.0.1-beta',
                        'websitePort': '8000',
                        'backendPort': '6848',
                        'tensorboardPort': '6006',
                        'trainPath': '/Robustar2/dataset/train',
                        'testPath': '/Robustar2/dataset/test',
                        'checkPointPath': '/Robustar2/checkpoint_images',
                        'influencePath': '/Robustar2/influence_images',
                        'configFile': 'configs.json'
                        }
        self.loadPath = ''
        self.savePath = ''

        self.firstTimeCheck = True


        # Match the corresponding signals and slots
        self.ui.nameInput.textEdited.connect(self.changeContainerName)
        self.ui.versionComboBox.currentIndexChanged.connect(self.changeImageVersion)
        self.ui.websitePortInput.textEdited.connect(self.changeWebsitePort)
        self.ui.backendPortInput.textEdited.connect(self.changeBackendPort)
        self.ui.tensorboardPortInput.textEdited.connect(self.changeTensorboardPort)
        self.ui.trainPathButton.clicked.connect(self.chooseTrainPath)
        self.ui.testPathButton.clicked.connect(self.chooseTestPath)
        self.ui.checkPointPathButton.clicked.connect(self.chooseCheckPointPath)
        self.ui.influencePathButton.clicked.connect(self.chooseInfluencePath)

        self.ui.loadConfigButton.clicked.connect(self.loadConfig)
        self.ui.saveConfigButton.clicked.connect(self.saveConfig)
        self.ui.startServerButton.clicked.connect(self.startServer)
        self.ui.stopServerButton.clicked.connect(self.stopServer)
        # self.ui.deleteServerButton.clicked.connect(self.deleteServer)

        self.ui.tabWidget.currentChanged.connect(self.renderContanierList)

        self.customSignals.printMessageSignal.connect(self.printMessage)
        self.customSignals.addItemSignal.connect(self.addItem)
        self.customSignals.removeItemSignal.connect(self.removeItem)

    def changeContainerName(self):
        self.configs['containerName'] = self.ui.nameInput.text()

    def changeImageVersion(self):
        self.configs['imageVersion'] = self.ui.versionComboBox.currentText()

    def changeWebsitePort(self):
        self.configs['websitePort'] = self.ui.websitePortInput.text()

    def changeBackendPort(self):
        self.configs['backendPort'] = self.ui.backendPortInput.text()

    def changeTensorboardPort(self):
        self.configs['tensorboardPort'] = self.ui.tensorboardPortInput.text()

    def chooseTrainPath(self):
        self.configs['trainPath'] = QFileDialog.getExistingDirectory(self, "Choose Train Set Path", self.cwd)
        self.ui.trainPathDisplay.setText(self.configs['trainPath'])

    def chooseTestPath(self):
        self.configs['testPath'] = QFileDialog.getExistingDirectory(self, "Choose Test Set Path", self.cwd)
        self.ui.testPathDisplay.setText(self.configs['testPath'])

    def chooseCheckPointPath(self):
        self.configs['checkPointPath'] = QFileDialog.getExistingDirectory(self, "Choose Check Points Path", self.cwd)
        self.ui.checkPointPathDisplay.setText(self.configs['checkPointPath'])

    def chooseInfluencePath(self):
        self.configs['influencePath'] = QFileDialog.getExistingDirectory(self, "Choose Influence Result Path", self.cwd)
        self.ui.influencePathDisplay.setText(self.configs['influencePath'])

    def loadConfig(self):
        self.loadPath, _ = QFileDialog.getOpenFileName(self, "Load Configs", self.cwd, "JSON Files (*.json);;All Files (*)")
        try:
            with open(self.loadPath, 'r') as f:
                self.configs = json.load(f)

                # Update the UI according to the loaded file
                self.ui.nameInput.setText(self.configs['containerName'])
                self.ui.versionComboBox.setCurrentText(self.configs['imageVersion'])
                self.ui.websitePortInput.setText(self.configs['websitePort'])
                self.ui.backendPortInput.setText(self.configs['backendPort'])
                self.ui.tensorboardPortInput.setText(self.configs['tensorboardPort'])
                self.ui.trainPathDisplay.setText(self.configs['trainPath'])
                self.ui.testPathDisplay.setText(self.configs['testPath'])
                self.ui.checkPointPathDisplay.setText(self.configs['checkPointPath'])
                self.ui.influencePathDisplay.setText(self.configs['influencePath'])

        except FileNotFoundError:
            print('Load path not found')

    def saveConfig(self):
        self.savePath, _ = QFileDialog.getOpenFileName(self, "Save Configs", self.cwd, "JSON Files (*.json);;All Files (*)")
        try:
            with open(self.savePath, 'w') as f:
                json.dump(self.configs, f)
        except FileNotFoundError:
            print('Save path not found')

    def startServer(self):

        image = 'paulcccccch/robustar:' + self.configs['imageVersion']

        def startServerInThread():
            try:
                # Get the container with the input name
                self.container = self.client.containers.get(self.configs['containerName'])

                # If the container has exited
                # Restart the container
                # Update both runningListWidget and exitedListWidget
                if self.container.status == 'exited':
                    self.container.restart()
                    self.customSignals.printMessageSignal.emit('Robustar is available at http://localhost:' + self.configs['websitePort'])
                    self.customSignals.addItemSignal.emit(self.ui.runningListWidget, self.configs['containerName'])
                    self.customSignals.removeItemSignal.emit(self.ui.exitedListWidget, self.configs['containerName'])

                # If the container is running
                elif self.container.status == 'running':
                    self.customSignals.printMessageSignal.emit('The server has already been running')

                # If the container is in other status
                else:
                    self.customSignals.printMessageSignal.emit('The server encountered an unexpected status')
                    time.sleep(5)
                    exit(1)

            # If the container with the input name has not been created yet
            # Create a new container and run it
            except docker.errors.NotFound:
                # If the version uses cuda
                # Set the device_requests parm
                if 'cuda' in image:
                    self.container = self.client.containers.run(
                        image,
                        detach=True,
                        name=self.configs['containerName'],
                        ports={
                            '80/tcp': (
                                '127.0.0.1', int(self.configs['websitePort'])),
                            '8000/tcp': ('127.0.0.1', 6848),
                            '6006/tcp': ('127.0.0.1', 6006),
                        },
                        mounts=[
                            docker.types.Mount(target='/Robustar2/dataset/train',
                                               source=self.configs['trainPath'],
                                               type='bind'),
                            docker.types.Mount(target='/Robustar2/dataset/test',
                                               source=self.configs['testPath'],
                                               type='bind'),
                            docker.types.Mount(target='/Robustar2/influence_images',
                                               source=self.configs['influencePath'],
                                               type='bind'),
                            docker.types.Mount(
                                target='/Robustar2/checkpoint_images ',
                                source=self.configs['checkPointPath'],
                                type='bind'),
                        ],
                        volumes=[
                            self.configs['configFile'] + ':/Robustar2/configs.json'],
                        device_requests=[
                            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                        ]
                    )
                    self.customSignals.printMessageSignal.emit('Robustar is available at http://localhost:' + self.configs['websitePort'])
                    self.customSignals.addItemSignal.emit(self.ui.runningListWidget, self.configs['containerName'])
                # If the version only uses cpu
                else:
                    self.container = self.client.containers.run(
                        image,
                        detach=True,
                        name=self.configs['containerName'],
                        ports={
                            '80/tcp': (
                                '127.0.0.1', int(self.configs['websitePort'])),
                            '8000/tcp': ('127.0.0.1', 6848),
                            '6006/tcp': ('127.0.0.1', 6006),
                        },
                        mounts=[
                            docker.types.Mount(target='/Robustar2/dataset/train',
                                               source=self.configs['trainPath'],
                                               type='bind'),
                            docker.types.Mount(target='/Robustar2/dataset/test',
                                               source=self.configs['testPath'],
                                               type='bind'),
                            docker.types.Mount(target='/Robustar2/influence_images',
                                               source=self.configs['influencePath'],
                                               type='bind'),
                            docker.types.Mount(
                                target='/Robustar2/checkpoint_images ',
                                source=self.configs['checkPointPath'],
                                type='bind'),
                        ],
                        volumes=[
                            self.configs['configFile'] + ':/Robustar2/configs.json']
                    )
                    self.customSignals.printMessageSignal.emit('Robustar is available at http://localhost:' + self.configs['websitePort'])
                    self.customSignals.addItemSignal.emit(self.ui.runningListWidget, self.configs['containerName'])
            except docker.errors.APIError:
                self.customSignals.printMessageSignal.emit('The server encountered an error')
                time.sleep(5)
                exit(1)

        startServerThread = Thread(target=startServerInThread)
        startServerThread.start()


    def stopServer(self):

        def stopServerInThread():
            try:
                self.container = self.client.containers.get(self.configs['containerName'])
                self.container.stop()
                # If the container has been stopped
                if self.container.status == 'exited':
                    self.customSignals.printMessageSignal.emit('The server has already been stopped')
                # If the container is running
                # Stop the container
                # Update both runningListWidget and exitedListWidget
                elif self.container.status == 'running':
                    self.container.stop()
                    self.customSignals.printMessageSignal.emit('The server is now stopped')
                    self.customSignals.addItemSignal.emit(self.ui.exitedListWidget, self.configs['containerName'])
                    self.customSignals.removeItemSignal.emit(self.ui.runningListWidget, self.configs['containerName'])
                # If the container is in other status
                else:
                    self.customSignals.printMessageSignal.emit('The server encountered an unexpected status')
                    time.sleep(5)
                    exit(1)
            except docker.errors.NotFound:
                self.customSignals.printMessageSignal.emit('The server has not been created yet')
            except docker.errors.APIError:
                self.customSignals.printMessageSignal.emit('The server encountered an error')
                time.sleep(5)
                exit(1)

        stopServerThread = Thread(target=stopServerInThread)
        stopServerThread.start()

    def renderContanierList(self, index):
        # If the current tab widget is manageTab
        # List the containers
        if(index == 1 and self.firstTimeCheck == True):
            self.listContainer()
            self.firstTimeCheck = False

    def listContainer(self):

        def listContainerInThread():
            containerList = self.client.containers.list(all=True)
            for container in containerList:
                if ('paulcccccch/robustar:' in str(container.image)):
                    if (container.status == 'running'):
                        self.customSignals.addItemSignal.emit(self.ui.runningListWidget, container.name)
                    elif (container.status == 'exited'):
                        self.customSignals.addItemSignal.emit(self.ui.exitedListWidget, container.name)
                    else:
                        self.customSignals.printMessageSignal.emit('The server encountered an unexpected status')
                        time.sleep(5)
                        exit(1)

        listContainerThread = Thread(target=listContainerInThread())
        listContainerThread.start()

    def printMessage(self, message):
        self.ui.messageBrowser.append(message)
        self.ui.messageBrowser.ensureCursorVisible()

    def addItem(self, listWidget, name):
        listWidget.addItem(name)

    def removeItem(self, listWidget, name):
        items = listWidget.findItems(name, Qt.MatchExactly)
        item = items[0]
        row = listWidget.row(item)
        listWidget.takeItem(row)

app = QApplication([])
launcher = Launcher()

launcher.ui.setFixedSize(800, 680)
launcher.ui.show()
app.exec_()