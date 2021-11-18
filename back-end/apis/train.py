from objects.RServer import RServer
from flask import request
from utils.train import start_train

app = RServer.getServer().getFlaskApp()


@app.route('/train', methods=['POST'])
def start_training():
    """
    Takes in a training config. 
    The server will check the configs before start training.
    """

    print("Requested to training with the following configuration: ")
    json_data = request.get_json()
    configs = json_data['configs']
    print(configs)

    # Return error message if config is invalid
    check_result = check_configs(configs)
    if check_result != 0:
        return {"msg": "Invalid Configuration!", "code": check_result}

    # Try to start training thread
    app.configs = configs
    print("DEBUG: Training request received! Setting up training...")

    # TODO: Save this train_thread variable somewhere. 
    # When a stop API is called, stop this thread.
    train_thread = start_train(app.configs)

    # Return error if training cannot be started
    if not train_thread:
        return {"msg": "Failed", "code": -1}

    # Training started succesfully!
    return {"msg": "Training started!", "code": 0}



def check_configs(config):
    """
    Check the config of the server. Returns 0 if config is valid. 
    Otherwise, return an error code from the following table:
    error code  |       meaning
        10      | Training set not found or not valid
        11      | Test set not found or not valid
        12      | Dev set not found or not valid
        13      | Class file not found or not valid
        14      | Weight file not found or not valid
        15      | Path for the source data set to be mirrored is not valid
        16      | User edit json file path not valid

        20      | paired train reg coeff not valid
        21      | learn rate not valid
        22      | epoch num not valid
        23      | image size not valid
        24      | thread number not valid
        25      | batch size not valid
    """
    # TODO: check the config here
    return 0

