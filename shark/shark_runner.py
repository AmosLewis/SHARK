# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from shark.torch_mlir_utils import get_torch_mlir_module, export_module_to_mlir_file
from shark.iree_utils import get_results, get_iree_compiled_module, export_iree_module_to_vmfb
from shark.functorch_utils import AOTModule
import torch
import argparse
import os
# from functorch_utils import AOTModule

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
              
class SharkRunner:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool,
        device: str,
        tracing_required: bool,
        from_aot: bool,
    ):
        self.parser = argparse.ArgumentParser(description='SHARK runner.')
        self.parser.add_argument("--repro_dir", help="Directory to which module files will be saved for reproduction or debugging.", type=dir_path, default="/tmp/")
        self.parser.add_argument("--save_mlir", default=False, action="store_true", help="Saves input MLIR module to /tmp/ directory.")
        self.parser.add_argument("--save_vmfb", default=False, action="store_true", help="Saves iree .vmfb module to /tmp/ directory.")
        self.parser.parse_args(namespace=self)
        self.torch_module = model
        self.input = input
        self.torch_mlir_module = get_torch_mlir_module(
            model, input, dynamic, tracing_required, from_aot
        )
        if self.save_mlir:
            export_module_to_mlir_file(self.torch_mlir_module, self.repro_dir)
        if self.save_vmfb:
            export_iree_module_to_vmfb(self.torch_mlir_module, device, self.repro_dir)
        (
            self.iree_compilation_module,
            self.iree_config,
        ) = get_iree_compiled_module(self.torch_mlir_module, device)
        
    def forward(self, input):
        return get_results(
            self.iree_compilation_module, input, self.iree_config
        )


class SharkInference:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = "cpu",
        jit_trace: bool = False,
        from_aot: bool = False,
        custom_inference_fn=None,
    ):
        self.model = model
        self.input = input
        self.from_aot = from_aot

        # if from_aot:
            # aot_module = AOTModule(
                # model, input, custom_inference_fn=custom_inference_fn
            # )
            # aot_module.generate_inference_graph()
            # self.model = aot_module.forward_graph
            # self.input = aot_module.forward_inputs

        self.shark_runner = SharkRunner(
            self.model, self.input, dynamic, device, jit_trace, from_aot
        )

    def forward(self, inputs):
        # TODO Capture weights and inputs in case of AOT, Also rework the
        # forward pass.
        inputs = self.input if self.from_aot else inputs
        input_list = [x.detach().numpy() for x in inputs]
        return self.shark_runner.forward(input_list)


class SharkTrainer:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        label: tuple,
        dynamic: bool = False,
        device: str = "cpu",
        jit_trace: bool = False,
        from_aot: bool = True,
    ):

        self.model = model
        self.input = input
        self.label = label
        aot_module = AOTModule(model, input, label)
        aot_module.generate_training_graph()
        self.forward_graph = aot_module.forward_graph
        self.forward_inputs = aot_module.forward_inputs
        self.backward_graph = aot_module.backward_graph
        print(self.backward_graph.graph)
        self.backward_inputs = aot_module.backward_inputs
        self.params = self.model.parameters()

    def train(self, input, optimizer=None):
        if optimizer is None:
            optimizer = torch.optim.SGD(self.params, .001)

        forward_inputs = []
        backward_inputs = [] #will these be labels?
        for input in self.forward_inputs:
            forward_inputs.append(input.detach().numpy())
        for input in self.backward_inputs:
            backward_inputs.append(input.detach().numpy())
#
        self.shark_forward = SharkRunner(
            self.forward_graph,
            forward_inputs,
            dynamic=False,
            device="cpu",
            tracing_required=False,
            from_aot=True,
            )
        
        self.shark_backward = SharkRunner(
            self.backward_graph,
            backward_inputs,
            dynamic=False,
            device="cpu",
            tracing_required=False,
            from_aot=True,
        )
#        # TODO: Pass the iter variable, and optimizer.
        iters = 1
        index = 0

        for it in range(iters):
            self.shark_forward.forward(forward_inputs)
            gradients = self.shark_backward.forward(backward_inputs)
            for p in self.params():
                p.grad = gradients[i]
                i+=1

        optimizer.step()
        return
