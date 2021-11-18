'''
Author: Chonghan Chen (paulcccccch@gmail.com)
-----
Last Modified: Thursday, 18th November 2021 12:18:49 am
Modified By: Chonghan Chen (paulcccccch@gmail.com)
-----
'''

from .RDataManager import RDataManager
from flask import Flask


# Wrapper for flask server instance
class RServer:

    serverInstance = None

    # Use createServer method instead!
    def __init__(self, datasetPath='/Robustar2/dataset',):
    
        app = Flask(__name__, )
        app.after_request(self.afterRequest)

        self.datasetPath = datasetPath
        self.app = app
        self.dataManager = RDataManager(datasetPath)

    @staticmethod
    def createServer(datasetPath):
        if RServer.serverInstance is None:
            RServer.serverInstance = RServer(datasetPath)
        else:
            assert datasetPath == RServer.serverInstance.datasetPath, \
            'Attempting to recreate an existing server with a different datapath'
        return RServer.serverInstance

    
    @staticmethod
    def getServer():
        return RServer.serverInstance
        
    
    def afterRequest(self, resp):
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Headers'] = '*'
        return resp

    def getFlaskApp(self):
        return self.app


    def run(self, port=8000, host='0.0.0.0', debug=True):
        self.port = port
        self.host = host
        self.debug = debug
        self.app.run(port=port, host=host, debug=debug)