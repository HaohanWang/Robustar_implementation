import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml import PairedDataset

import pytest


class Test1:
    def test_train_paired_train_data_success(self):
        # manually creates the the first paired image which is filled with (255,255,255)
        paired_img_path = "/Robustar2/dataset/paired/bird/0.JPEG"
        encoded_string = "iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAYAAAAaLWrhAAAAAXNSR0IArs4c6QAABQxJREFUeF7t08EJADAMA7F2/6Fd6BL3URYwiNzdtuMIEEgErgATd6MEvoAAPQKBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimyYgQD9AIBQQYIhvmoAA/QCBUECAIb5pAgL0AwRCAQGG+KYJCNAPEAgFBBjimybwAPj3fY6Bc5rWAAAAAElFTkSuQmCC"
        decoded = base64.b64decode(encoded_string)
        with Image.open(BytesIO(decoded)) as img:
            to_save = img.resize((224, 224))
            to_save.save(paired_img_path, format='png')

        # test it through the framework
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5),
            #                      std=(0.5, 0.5, 0.5))
        ])

        mixture_methods = ['mixture', 'pure_black', 'noise', 'noise_weak', 'noise_minor',
                           'random_pure', 'hstrips', 'vstrips']

        for method in mixture_methods:
            train_set = PairedDataset('/Robustar2/dataset/train', '/Robustar2/dataset/paired', 32,
                                      transform, None, method, False)
            data_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False,
                                                      num_workers=1)

            img = 0
            for idx, data in enumerate(data_loader):
                if idx in [0]:
                    # img = data[0][0].squeeze(0).permute(1, 2, 0).numpy() # train image
                    img = data[1][0].squeeze(0).permute(1, 2, 0).numpy()  # paired image
                    img = transform(img)
                    img = np.swapaxes(img, 0, 1)
                    img = np.swapaxes(img, 1, 2)
                    # print(img)
                    plt.imshow(img.numpy())
                    plt.show()
                else:
                    break

            assert self._test_image(np.array(img), method)

    def _test_image(self, img_data, mixture_method):
        if mixture_method == 'pure_black':
            bool_arr = (img_data == np.full((1, 3), 0))
            result = np.all(bool_arr)
            return result
        elif mixture_method == 'random_pure':
            bool_arr = (img_data == img_data[0][0])
            result = np.all(bool_arr)
            return result

        # TODO: other mexture methods

        # TODO: other test cases

        return True
