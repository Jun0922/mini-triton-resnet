import sys
import time
import random

import tritonclient.grpc as grpcclient
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tritonclient.utils import *


model_name = "resnet"
DATASET_PATH = "/data"

acc_infer_time_ms = 0

with grpcclient.InferenceServerClient("mini-triton-server-service:8001") as client:
    test_set = MNIST(
        root=DATASET_PATH, train=False, download=False,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    inf_results = []
    for i, (image, label) in enumerate(test_loader):
        gt = label.item()

        input0_data = image.numpy()
        inputs = [
            grpcclient.InferInput(
                "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
            ),
        ]

        inputs[0].set_data_from_numpy(input0_data)

        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT0"),
        ]

        # infer
        s_time = time.time()
        response = client.infer(model_name, inputs, request_id=str(i), outputs=outputs)
        e_time = time.time()
        elapsed_time_ms = int((e_time - s_time) * 1000)

        acc_infer_time_ms += elapsed_time_ms

        if (i + 1) % 100 == 0:
            print(f"avg inference time | {int(acc_infer_time_ms / 100)} ms")
            print(f"{int(((i + 1) / len(test_loader)) * 100)}% iter done. (cur, total) | {(i + 1, len(test_loader))}")
            acc_infer_time_ms = 0

        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0")
        output = output0_data.item()

        inf_results.append((gt, output))

    random_samples = random.sample(inf_results, 10)
    acc_cnt = 0
    for i, (gt, output)in enumerate(random_samples):
        if gt == output:
            print(f"correct! | (gt, output) | {(gt, output)}")
            acc_cnt += 1
        else:
            print(f"incorrect! | (gt, output) | {(gt, output)}")

    print(f"accuracy | {int(acc_cnt / 10 * 100)}%")

    sys.exit(0)