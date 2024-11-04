from model_utils import train_model


if __name__ == '__main__':
    resnet_model, resnet_results = train_model(
        model_name="ResNet",
        model_hparams={
            "num_classes": 10, "c_hidden": [16, 32, 64],
            "num_blocks": [3, 3, 3], "act_fn_name": "relu"
        },
        optimizer_name="SGD",
        optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
    )
