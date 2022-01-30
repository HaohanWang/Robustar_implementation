from objects.RServer import RServer
from objects.RResponse import RResponse
from flask import request
from io import BytesIO
import numpy as np
from PIL import Image
import base64
from utils.image_utils import imageURLToPath, get_train_from_annotated
from utils.path_utils import get_paired_path

server = RServer.getServer()
app = server.getFlaskApp()
dataManager = server.getDataManager()

@app.route('/edit/<split>/<image_id>', methods=['POST'])
def user_edit(split, image_id):
    # TODO: Maybe support editing other splits as well? Or not?
    if split not in ['train', 'annotated']:
        raise NotImplemented('Split {} not supported! Currently we only support editing the `train` or `annotated` splits!'.format(split))

    if split == 'annotated':
        image_id = get_train_from_annotated(image_id)
        split = 'train'

    json_data = request.get_json()
    encoded_string = json_data['image'].split(',')[1]
    decoded = base64.b64decode(encoded_string)

    h = int(json_data['image_height'])
    w = int(json_data['image_width'])
    with Image.open(BytesIO(decoded)) as img:
        
        img_path = imageURLToPath('{}/{}'.format(split, image_id))

        paired_img_path = get_paired_path(img_path, dataManager.train_root, dataManager.paired_root)

        to_save = img.resize((w, h))
        to_save = to_save.convert('RGB') # image comming from canvas is RGBA

        to_save.save(paired_img_path)

        if int(image_id) in dataManager.annotatedInvBuffer:
            save_idx = dataManager.annotatedInvBuffer[int(image_id)]
        else:
            save_idx = len(dataManager.annotatedBuffer)
            dataManager.annotatedInvBuffer[int(image_id)] = save_idx
        dataManager.annotatedBuffer[save_idx] = int(image_id)

        dataManager.dump_annotated_list() # TODO: Change this to SQLite

    return RResponse.ok("Success!")


if __name__ == '__main__':
    print(RServer)