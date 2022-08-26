# import faulthandler
# faulthandler.enable()
import torch
import torch_mlir
from transformers import AutoModelForSequenceClassification

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import tempfile

class HuggingFaceLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bigscience/bloom-560m",  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]

torch.manual_seed(0)

model = HuggingFaceLanguage()
test_input = torch.randint(2, (1, 128))
actual_out = model(test_input)

# import numpy as np
# test_input_ny = test_input.detach().numpy()
# input_tuple = (test_input_ny,)
# np.savez('inputs.npz', *input_tuple)
# output_ny = actual_out.detach().numpy()
# output_tuple = (output_ny,)
# np.savez('golden_out.npz', *output_tuple)

# print(actual_out)

fx_g = make_fx(
        model,
        decomposition_table=get_decompositions(
            [
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
            ]
        ),
    )(
            test_input
    )

# print(fx_g.graph)

fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
fx_g.recompile()

def strip_overloads(gm):
    """ 
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()

strip_overloads(fx_g)

ts_g = torch.jit.script(fx_g)
# print(ts_g.graph)

# temp = tempfile.NamedTemporaryFile(
#     suffix="_shark_ts", prefix="temp_ts_"
# )
# ts_g.save(temp.name)
# new_ts = torch.jit.load(temp.name)
# print (ts_g.graph)

module = torch_mlir.compile(
    ts_g, [test_input], torch_mlir.OutputType.LINALG_ON_TENSORS, use_tracing=True, verbose=False
)

# module.dump()

from shark.shark_inference import SharkInference


mlir_model = module
func_name = "forward"

shark_module = SharkInference(
    mlir_model, func_name, device="cpu", mlir_dialect="tm_tensor"
)
shark_module.compile()

def shark_result(x):
    x_ny = x.detach().numpy()
    inputs = (x_ny,)
    result = shark_module.forward(inputs)
    return torch.from_numpy(result)

observed_out = shark_result(test_input)

print(actual_out, observed_out)