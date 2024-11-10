
import json
import os

import torch
import triton_python_backend_utils as pb_utils
import numpy as np
import torch.nn.functional as F

from CIFARModule import CIFARModule


MODEL_PATH = "./best_resnet.ckpt"


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        # Instantiate the PyTorch model
        ckpt_file = os.path.join(MODEL_PATH)
        self.resnet = CIFARModule.load_from_checkpoint(ckpt_file)
        print("Resnet Initialized...", flush=True)

    def execute(self, requests):
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_tensor_0 = torch.from_numpy(in_0.as_numpy())

            # infer output
            out_tensor_0 = self.resnet(in_tensor_0)

            # find output class
            softmax_probs = F.softmax(out_tensor_0, dim=1)
            predicted_label = torch.argmax(softmax_probs, dim=1).item()

            # convert to triton tensor
            result_np = np.array(predicted_label)
            result = pb_utils.Tensor("OUTPUT0", result_np)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[result]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")