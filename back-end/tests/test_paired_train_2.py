import json

import pytest

from objects.RServer import RServer
from objects.RTask import RTask, TaskType
from server import start_server

import os
import os.path as osp


# start_server()
# def test_valid_app_and_server():
#     server = RServer.getServer()
#     assert server
#     assert server.getFlaskApp()
#     # assert server.dataManager
#     # assert RServer.getModelWrapper()


@pytest.fixture()
def server():
    _cleanup()
    start_server()
    server_r = RServer.getServer()
    yield server_r


def _cleanup():
    base_dir = osp.join('/', 'Robustar2').replace('\\', '/')
    dataset_dir = osp.join(base_dir, 'dataset').replace('\\', '/')
    test_correct_root = osp.join(dataset_dir, 'test_correct.txt').replace('\\', '/')
    test_incorrect_root = osp.join(dataset_dir, 'test_incorrect.txt').replace('\\', '/')
    validation_correct_root = osp.join(dataset_dir, 'validation_correct.txt').replace('\\', '/')
    validation_incorrect_root = osp.join(dataset_dir, 'validation_incorrect.txt').replace('\\', '/')
    annotated_root = osp.join(dataset_dir, 'annotated.txt').replace('\\', '/')
    paired_root = osp.join(dataset_dir, 'paired').replace('\\', '/')
    if osp.exists(test_correct_root):
        print("cleanup > delete " + test_correct_root)
        os.remove(test_correct_root)
    if osp.exists(test_incorrect_root):
        print("cleanup > delete " + test_incorrect_root)
        os.remove(test_incorrect_root)
    if osp.exists(validation_correct_root):
        print("cleanup > delete " + validation_correct_root)
        os.remove(validation_correct_root)
    if osp.exists(validation_incorrect_root):
        print("cleanup > delete " + validation_incorrect_root)
        os.remove(validation_incorrect_root)
    if osp.exists(annotated_root):
        print("cleanup > delete " + annotated_root)
        os.remove(annotated_root)
    if osp.exists(paired_root):
        print("cleanup > delete " + paired_root)
        for subfolder in os.listdir(paired_root):
            subfolder_root = osp.join(paired_root, subfolder).replace('\\', '/')
            print("cleanup >> delete " + subfolder_root)
            for image in os.listdir(subfolder_root):
                image_root = osp.join(subfolder_root, image).replace('\\', '/')
                if os.path.isfile(image_root):
                    # print("cleanup >>> delete " + image_root)
                    os.remove(image_root)
            os.rmdir(subfolder_root)
        os.rmdir(paired_root)


class Test2:
    def test_config_success(self, server):
        app = server.getFlaskApp()
        app.config['TESTING'] = True
        client = app.test_client()
        rv = client.get("/config").get_json()
        assert rv['code'] == 0
        assert rv['data'] == {
            "weight_to_load": "resnet-18.pth",
            "model_arch": "resnet-18-32x32",
            "device": "cpu",
            "pre_trained": False,
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 8,
            "image_size": 32,
            "image_padding": "none",
            "num_classes": 9
        }
        app.config['TESTING'] = False

    def test_train_paired_train_data_success(self, server):
        app = server.getFlaskApp()
        app.config['TESTING'] = True
        client = app.test_client()

        data = {"configs": {
            "weight_to_load": "resnet-18.pth",
            "model_arch": "resnet-18-32x32",
            "device": "cpu",
            "pre_trained": False,
            "batch_size": 16,
            "shuffle": True,
            "num_workers": 8,
            "image_size": 32,
            "image_padding": "none",
            "num_classes": 9,
            "learn_rate": 0.1,
            "thread": 8,
            "save_dir": "/Robustar2/checkpoints",
            'use_paired_train': 'yes',
            'paired_data_path': '/Robustar2/dataset/paired',
            'paired_train_reg_coeff': 0.001,
            'mixture': 'random_pure',
            'class_path': './model/cifar-class.txt',
            'train_path': '/Robustar2/dataset/train',
            'test_path': '/Robustar2/dataset/test',
            'paired_data_path': '/Robustar2/dataset/paired',
            'user_edit_buffering': False,
            'epoch': 8,
            'auto_save_model': 'yes'
        }
        }
        rv = client.post("/train", json=json.loads(json.dumps(data))).get_json()
        assert rv['code'] == 0
        sc = server.getServerConfigs()
        assert sc == {'weight_to_load': 'resnet-18.pth', 'model_arch': 'resnet-18-32x32',
                      'device': 'cpu', 'pre_trained': False, 'batch_size': 16, 'shuffle': True,
                      'num_workers': 8, 'image_size': 32, 'image_padding': 'none', 'num_classes': 9}

        # RTask.exit_tasks_of_type(TaskType.Training) # TODO : not working

        app.config['TESTING'] = False
