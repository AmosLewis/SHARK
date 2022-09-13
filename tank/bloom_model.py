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

############# 31-43copy from squeezenet_lockstep.py#50 #############

from shark.torch_mlir_lockstep_tensor import TorchMLIRLockstepTensor
input_detached_clone = test_input.clone()
eager_input_batch = TorchMLIRLockstepTensor(input_detached_clone)
print("getting torch-mlir result")
output = model(eager_input_batch) # RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
print("output: ", output)
static_output = output.elem
print("static_output: ", static_output)
shark_out = static_output[0]
print("The obtained result via shark is: ", shark_out)
print("The golden result is:", actual_out)

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
# tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[3.5580e-33, 0.0000e+00]])



# ➜  SHARK git:(bloom) ✗ python tank/bloom_model.py
# Some weights of BloomForSequenceClassification were not initialized from the model checkpoint at bigscience/bloom-560m and are newly initialized: ['score.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# getting torch-mlir result
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:180: UserWarning: Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 107, in __torch_dispatch__
#     raise UnsupportedByTorchMlirEagerMode(op_name)
# torch_mlir.eager_mode.torch_mlir_dispatch.UnsupportedByTorchMlirEagerMode: view.default
#
#   warnings.warn(traceback.format_exc())
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:182: UserWarning: Couldn't use TorchMLIR eager because current incompatibility: *view.default*; running through PyTorch eager.
#   warnings.warn(
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:180: UserWarning: Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 132, in __torch_dispatch__
#     assert len(eager_module.body.operations[0].arguments) == len(
# AssertionError: Number of parameters and number of arguments differs.
#
#   warnings.warn(traceback.format_exc())
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:186: UserWarning: Couldn't use TorchMLIR eager because of error: *Number of parameters and number of arguments differs.*; Running through PyTorch eager
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:180: UserWarning: Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 135, in __torch_dispatch__
#     op_mlir_backend_callable = backend.compile(eager_module)
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/iree_eager_backend.py", line 68, in compile
#     run_pipeline_with_repro_report(
#   File "/home/chi/src/ubuntu20/shark/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir/torch_mlir/compiler_utils.py", line 73, in run_pipeline_with_repro_report
#     raise TorchMlirCompilerError(trimmed_message) from None
# torch_mlir.compiler_utils.TorchMlirCompilerError: EagerMode failed with the following diagnostics:
# error: unsupported by backend contract: tensor with unknown rank
# note: see current operation: %2 = "torch.tensor_static_info_cast"(%arg0) : (!torch.vtensor<[1,128,16,192],f32>) -> !torch.vtensor<*,f32>
# note: this is likely due to a missing shape transfer function in shape_lib_gen.py
#
#
# Error can be reproduced with:
# $ torch-mlir-opt -pass-pipeline='torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline' /tmp/graphinputTensor2intprimConstantvalue643intprimConstantvalue10Tensoratensplitinput23return0.mlir
# Add '-mlir-print-ir-after-all -mlir-disable-threading' to get the IR dump for debugging purpose.
#
#
#   warnings.warn(traceback.format_exc())
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:186: UserWarning: Couldn't use TorchMLIR eager because of error: *EagerMode failed with the following diagnostics:
# error: unsupported by backend contract: tensor with unknown rank
# note: see current operation: %2 = "torch.tensor_static_info_cast"(%arg0) : (!torch.vtensor<[1,128,16,192],f32>) -> !torch.vtensor<*,f32>
# note: this is likely due to a missing shape transfer function in shape_lib_gen.py
#
#
# Error can be reproduced with:
# $ torch-mlir-opt -pass-pipeline='torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline' /tmp/graphinputTensor2intprimConstantvalue643intprimConstantvalue10Tensoratensplitinput23return0.mlir
# Add '-mlir-print-ir-after-all -mlir-disable-threading' to get the IR dump for debugging purpose.
# *; Running through PyTorch eager
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131008 / 131072 (100%)
# Max absolute difference: 4.236016
# Max relative difference: 224.99109
#  x: array([[[-0.148702, -0.148702, -0.148702, ..., -0.148702, -0.148702,
#          -0.148702],
#         [-0.148702, -0.148702, -0.148702, ..., -0.148702, -0.148702,...
#  y: array([[[-0.148702,  0.985857, -0.642777, ..., -0.26846 ,  0.077838,
#           0.352737],
#         [-0.148702,  0.985857, -0.642777, ..., -0.26846 ,  0.077838,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131008 / 131072 (100%)
# Max absolute difference: 24.35872
# Max relative difference: 187.21574
#  x: array([[[-0.262711, -0.262711, -0.262711, ..., -0.262711, -0.262711,
#          -0.262711],
#         [-0.262711, -0.262711, -0.262711, ..., -0.262711, -0.262711,...
#  y: array([[[-2.627107e-01, -2.627107e-01, -2.627107e-01, ...,
#          -2.627107e-01, -2.627107e-01, -2.627107e-01],
#         [-4.661741e-01, -4.661741e-01, -4.661741e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:180: UserWarning: Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 126, in __torch_dispatch__
#     eager_module = build_mlir_module(func, normalized_kwargs)
#   File "/home/chi/src/ubuntu20/shark/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir/torch_mlir/eager_mode/ir_building.py", line 343, in build_mlir_module
#     assert len(annotations) == len(
# AssertionError: Number of annotations and number of graph inputs differs.
#
#   warnings.warn(traceback.format_exc())
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:186: UserWarning: Couldn't use TorchMLIR eager because of error: *Number of annotations and number of graph inputs differs.*; Running through PyTorch eager
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:180: UserWarning: Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 107, in __torch_dispatch__
#     raise UnsupportedByTorchMlirEagerMode(op_name)
# torch_mlir.eager_mode.torch_mlir_dispatch.UnsupportedByTorchMlirEagerMode: detach.default
#
#   warnings.warn(traceback.format_exc())
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:182: UserWarning: Couldn't use TorchMLIR eager because current incompatibility: *detach.default*; running through PyTorch eager.
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 130944 / 131072 (99.9%)
# Max absolute difference: 2.801336
# Max relative difference: 889.7943
#  x: array([[[0.349916, 0.349916, 0.349916, ..., 0.349916, 0.349916,
#          0.349916],
#         [0.349916, 0.349916, 0.349916, ..., 0.349916, 0.349916,...
#  y: array([[[ 0.349916,  0.351781,  0.981228, ..., -0.033237,  0.440027,
#          -0.385875],
#         [ 0.349916,  0.351781,  0.981228, ..., -0.033237,  0.440027,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 2.3344822
# Max relative difference: 196455.86
#  x: array([[[0.349916, 0.349916, 0.349916, ..., 0.349916, 0.349916,
#          0.349916],
#         [0.349916, 0.349916, 0.349916, ..., 0.349916, 0.349916,...
#  y: array([[[ 0.349916,  0.351781,  0.981228, ...,  0.366801, -0.06825 ,
#          -0.012859],
#         [ 0.349916,  0.351781,  0.981228, ...,  0.366801, -0.06825 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 2.3344822
# Max relative difference: 196455.86
#  x: array([[0.349916, 0.349916, 0.349916, ..., 0.349916, 0.349916, 0.349916],
#        [0.349916, 0.349916, 0.349916, ..., 0.349916, 0.349916, 0.349916],
#        [0.349916, 0.349916, 0.349916, ..., 0.349916, 0.349916, 0.349916],...
#  y: array([[ 0.349916,  0.351781,  0.981228, ...,  0.366801, -0.06825 ,
#         -0.012859],
#        [ 0.349916,  0.351781,  0.981228, ...,  0.366801, -0.06825 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:180: UserWarning: Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 107, in __torch_dispatch__
#     raise UnsupportedByTorchMlirEagerMode(op_name)
# torch_mlir.eager_mode.torch_mlir_dispatch.UnsupportedByTorchMlirEagerMode: _unsafe_view.default
#
#   warnings.warn(traceback.format_exc())
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:182: UserWarning: Couldn't use TorchMLIR eager because current incompatibility: *_unsafe_view.default*; running through PyTorch eager.
#   warnings.warn(
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:200: UserWarning: Found aliased arg, but didn't copy tensor contents. This could lead to incorrect results for E2E model execution but doesn't affect the validity of the lockstep op verification.
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524226 / 524288 (100%)
# Max absolute difference: 3.4689276
# Max relative difference: 39209.996
#  x: array([[-0.017029, -0.017029, -0.017029, ..., -0.017029, -0.017029,
#         -0.017029],
#        [-0.017029, -0.017029, -0.017029, ..., -0.017029, -0.017029,...
#  y: array([[-0.017029, -0.148383, -0.103539, ..., -0.169963, -0.023132,
#         -0.165966],
#        [-0.017029, -0.148383, -0.103539, ..., -0.169963, -0.023132,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 6.404586
# Max relative difference: 7458.912
#  x: array([[[0.145027, 0.145027, 0.145027, ..., 0.145027, 0.145027,
#          0.145027],
#         [0.145027, 0.145027, 0.145027, ..., 0.145027, 0.145027,...
#  y: array([[[ 0.145027, -0.199546, -0.217516, ...,  0.223763, -0.395327,
#          -0.168794],
#         [ 0.145027, -0.199546, -0.217516, ...,  0.223763, -0.395327,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 51.258434
# Max relative difference: 15753.126
#  x: array([[[-0.116204, -0.116204, -0.116204, ..., -0.116204, -0.116204,
#          -0.116204],
#         [-0.116204, -0.116204, -0.116204, ..., -0.116204, -0.116204,...
#  y: array([[[-1.162036e-01, -1.162037e-01, -1.162036e-01, ...,
#          -2.356637e-02, -8.510312e-02, -1.291886e-01],
#         [ 6.809323e-01,  6.809323e-01,  6.809324e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131062 / 131072 (100%)
# Max absolute difference: 2.865023
# Max relative difference: 1034447.3
#  x: array([[[-0.300582, -0.300582, -0.300582, ..., -0.300582, -0.300582,
#          -0.300582],
#         [-0.300582, -0.300582, -0.300582, ..., -0.300582, -0.300582,...
#  y: array([[[-0.300582,  0.544126,  0.009443, ...,  0.392907,  0.356756,
#           0.372702],
#         [-0.300582,  0.544127,  0.009443, ...,  0.392907,  0.356756,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131053 / 131072 (100%)
# Max absolute difference: 2.279744
# Max relative difference: 69298.266
#  x: array([[[-0.300582, -0.300582, -0.300582, ..., -0.300582, -0.300582,
#          -0.300582],
#         [-0.300582, -0.300582, -0.300582, ..., -0.300582, -0.300582,...
#  y: array([[[-0.300582,  0.544126,  0.009443, ..., -0.007198, -0.361515,
#          -0.495602],
#         [-0.300582,  0.544127,  0.009443, ..., -0.007198, -0.361516,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131053 / 131072 (100%)
# Max absolute difference: 2.279744
# Max relative difference: 69298.266
#  x: array([[-0.300582, -0.300582, -0.300582, ..., -0.300582, -0.300582,
#         -0.300582],
#        [-0.300582, -0.300582, -0.300582, ..., -0.300582, -0.300582,...
#  y: array([[-0.300582,  0.544126,  0.009443, ..., -0.007198, -0.361515,
#         -0.495602],
#        [-0.300582,  0.544127,  0.009443, ..., -0.007198, -0.361516,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524073 / 524288 (100%)
# Max absolute difference: 4.7676167
# Max relative difference: 1678136.2
#  x: array([[-0.156289, -0.156289, -0.156289, ..., -0.156289, -0.156289,
#         -0.156289],
#        [-0.156289, -0.156289, -0.156289, ..., -0.156289, -0.156289,...
#  y: array([[-0.156289, -0.145878, -0.082335, ...,  0.052713,  0.019069,
#         -0.044299],
#        [-0.156289, -0.145878, -0.082335, ...,  0.052713,  0.019069,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 7.3822107
# Max relative difference: 487963.5
#  x: array([[[-2.937568, -2.937568, -2.937568, ..., -2.937568, -2.937568,
#          -2.937568],
#         [-2.937568, -2.937568, -2.937568, ..., -2.937568, -2.937568,...
#  y: array([[[-2.937568,  0.77797 ,  2.115358, ...,  0.506081,  0.043954,
#           0.900285],
#         [-2.937568,  0.77797 ,  2.115358, ...,  0.506081,  0.043954,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 10.721197
# Max relative difference: 93634.164
#  x: array([[[-1.775795, -1.775795, -1.775795, ..., -1.775795, -1.775795,
#          -1.775795],
#         [-1.775795, -1.775795, -1.775795, ..., -1.775795, -1.775795,...
#  y: array([[[-1.775795e+00, -1.775795e+00, -1.775796e+00, ...,
#          -4.686096e-01, -4.320640e-01, -4.057743e-01],
#         [ 5.828291e-01,  5.828288e-01,  5.828288e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 2.3143802
# Max relative difference: 1979.4156
#  x: array([[[-0.042835, -0.042835, -0.042835, ..., -0.042835, -0.042835,
#          -0.042835],
#         [-0.042835, -0.042835, -0.042835, ..., -0.042835, -0.042835,...
#  y: array([[[-0.042835, -0.400479,  0.247279, ..., -0.054174,  0.199758,
#          -0.138755],
#         [-0.042835, -0.400479,  0.247279, ..., -0.054174,  0.199758,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 2.31438
# Max relative difference: 84476.67
#  x: array([[[-0.042835, -0.042835, -0.042835, ..., -0.042835, -0.042835,
#          -0.042835],
#         [-0.042835, -0.042835, -0.042835, ..., -0.042835, -0.042835,...
#  y: array([[[-0.042835, -0.400479,  0.247279, ...,  0.109486, -0.044048,
#           0.58589 ],
#         [-0.042835, -0.400479,  0.247279, ...,  0.109485, -0.044048,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 2.31438
# Max relative difference: 84476.67
#  x: array([[-0.042835, -0.042835, -0.042835, ..., -0.042835, -0.042835,
#         -0.042835],
#        [-0.042835, -0.042835, -0.042835, ..., -0.042835, -0.042835,...
#  y: array([[-0.042835, -0.400479,  0.247279, ...,  0.109486, -0.044048,
#          0.58589 ],
#        [-0.042835, -0.400479,  0.247279, ...,  0.109485, -0.044048,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524224 / 524288 (100%)
# Max absolute difference: 3.3418314
# Max relative difference: 88121.266
#  x: array([[-0.02101, -0.02101, -0.02101, ..., -0.02101, -0.02101, -0.02101],
#        [-0.02101, -0.02101, -0.02101, ..., -0.02101, -0.02101, -0.02101],
#        [-0.02101, -0.02101, -0.02101, ..., -0.02101, -0.02101, -0.02101],...
#  y: array([[-0.02101 , -0.141565, -0.169839, ..., -0.118222, -0.168672,
#         -0.165505],
#        [-0.02101 , -0.141565, -0.169839, ..., -0.118221, -0.168672,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131061 / 131072 (100%)
# Max absolute difference: 4.64468
# Max relative difference: 8420.842
#  x: array([[[-0.091449, -0.091449, -0.091449, ..., -0.091449, -0.091449,
#          -0.091449],
#         [-0.091449, -0.091449, -0.091449, ..., -0.091449, -0.091449,...
#  y: array([[[-9.144861e-02,  1.718178e-01, -3.751643e-01, ...,
#          -1.064263e+00,  7.343606e-01,  1.246592e-02],
#         [-9.144871e-02,  1.718182e-01, -3.751646e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 22.54736
# Max relative difference: 327004.56
#  x: array([[[-1.034244, -1.034244, -1.034244, ..., -1.034244, -1.034244,
#          -1.034244],
#         [-1.034244, -1.034244, -1.034244, ..., -1.034244, -1.034244,...
#  y: array([[[-1.034244e+00, -1.034244e+00, -1.034244e+00, ...,
#          -6.659001e-01, -6.832404e-01, -6.915847e-01],
#         [ 6.254903e-01,  6.254904e-01,  6.254904e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 3.3021083
# Max relative difference: 39633.504
#  x: array([[[-0.067292, -0.067292, -0.067292, ..., -0.067292, -0.067292,
#          -0.067292],
#         [-0.067292, -0.067292, -0.067292, ..., -0.067292, -0.067292,...
#  y: array([[[-0.067292,  0.465915,  0.07524 , ..., -0.076124, -0.334313,
#           0.093292],
#         [-0.067291,  0.465915,  0.075239, ..., -0.076124, -0.334313,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 2.7421422
# Max relative difference: 37352.957
#  x: array([[[-0.067292, -0.067292, -0.067292, ..., -0.067292, -0.067292,
#          -0.067292],
#         [-0.067292, -0.067292, -0.067292, ..., -0.067292, -0.067292,...
#  y: array([[[-0.067292,  0.465915,  0.07524 , ..., -0.120865,  0.258179,
#          -0.585785],
#         [-0.067291,  0.465915,  0.075239, ..., -0.120865,  0.258179,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 2.7421422
# Max relative difference: 37352.957
#  x: array([[-0.067292, -0.067292, -0.067292, ..., -0.067292, -0.067292,
#         -0.067292],
#        [-0.067292, -0.067292, -0.067292, ..., -0.067292, -0.067292,...
#  y: array([[-0.067292,  0.465915,  0.07524 , ..., -0.120865,  0.258179,
#         -0.585785],
#        [-0.067291,  0.465915,  0.075239, ..., -0.120865,  0.258179,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524276 / 524288 (100%)
# Max absolute difference: 8.325775
# Max relative difference: 842509.25
#  x: array([[0.242898, 0.242898, 0.242898, ..., 0.242898, 0.242898, 0.242898],
#        [0.242898, 0.242898, 0.242898, ..., 0.242898, 0.242898, 0.242898],
#        [0.242898, 0.242898, 0.242898, ..., 0.242898, 0.242898, 0.242898],...
#  y: array([[ 0.242898, -0.108406, -0.044345, ..., -0.15908 , -0.065063,
#         -0.159445],
#        [ 0.242898, -0.108406, -0.044345, ..., -0.15908 , -0.065063,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 5.0763364
# Max relative difference: 76508.08
#  x: array([[[0.942817, 0.942817, 0.942817, ..., 0.942817, 0.942817,
#          0.942817],
#         [0.942817, 0.942817, 0.942817, ..., 0.942817, 0.942817,...
#  y: array([[[ 0.942817,  1.333151, -0.648047, ..., -0.790512, -1.1147  ,
#           1.104549],
#         [ 0.942816,  1.33315 , -0.648047, ..., -0.790512, -1.1147  ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131061 / 131072 (100%)
# Max absolute difference: 15.785866
# Max relative difference: 13080.798
#  x: array([[[0.371349, 0.371349, 0.371349, ..., 0.371349, 0.371349,
#          0.371349],
#         [0.371349, 0.371349, 0.371349, ..., 0.371349, 0.371349,...
#  y: array([[[ 0.371349,  0.371349,  0.371349, ...,  0.129261,  0.130133,
#           0.137731],
#         [ 0.666184,  0.666185,  0.666185, ...,  1.246147,  1.200032,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 3.9329112
# Max relative difference: 64624.918
#  x: array([[[-0.112427, -0.112427, -0.112427, ..., -0.112427, -0.112427,
#          -0.112427],
#         [-0.112427, -0.112427, -0.112427, ..., -0.112427, -0.112427,...
#  y: array([[[-1.124269e-01, -1.122310e-01, -1.207412e-01, ...,
#           2.858531e-01, -2.113302e-01,  4.041998e-01],
#         [-1.124267e-01, -1.122308e-01, -1.207414e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 2.0023842
# Max relative difference: 91064.06
#  x: array([[[-0.112427, -0.112427, -0.112427, ..., -0.112427, -0.112427,
#          -0.112427],
#         [-0.112427, -0.112427, -0.112427, ..., -0.112427, -0.112427,...
#  y: array([[[-0.112427, -0.112231, -0.120741, ...,  0.456976, -0.10247 ,
#          -0.327618],
#         [-0.112427, -0.112231, -0.120741, ...,  0.456976, -0.10247 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 2.0023842
# Max relative difference: 91064.06
#  x: array([[-0.112427, -0.112427, -0.112427, ..., -0.112427, -0.112427,
#         -0.112427],
#        [-0.112427, -0.112427, -0.112427, ..., -0.112427, -0.112427,...
#  y: array([[-0.112427, -0.112231, -0.120741, ...,  0.456976, -0.10247 ,
#         -0.327618],
#        [-0.112427, -0.112231, -0.120741, ...,  0.456976, -0.10247 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524274 / 524288 (100%)
# Max absolute difference: 40.38826
# Max relative difference: 2209306.8
#  x: array([[0.427976, 0.427976, 0.427976, ..., 0.427976, 0.427976, 0.427976],
#        [0.427976, 0.427976, 0.427976, ..., 0.427976, 0.427976, 0.427976],
#        [0.427976, 0.427976, 0.427976, ..., 0.427976, 0.427976, 0.427976],...
#  y: array([[ 4.279764e-01, -1.241705e-01, -1.695486e-01, ...,  2.569507e-01,
#          1.567214e+00, -1.134349e-02],
#        [ 4.279764e-01, -1.241707e-01, -1.695486e-01, ...,  2.569507e-01,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 5.6343637
# Max relative difference: 476251.72
#  x: array([[[0.624511, 0.624511, 0.624511, ..., 0.624511, 0.624511,
#          0.624511],
#         [0.624511, 0.624511, 0.624511, ..., 0.624511, 0.624511,...
#  y: array([[[ 0.624511, -0.339705, -0.889052, ..., -0.439274,  0.11043 ,
#           0.708226],
#         [ 0.624511, -0.339705, -0.889052, ..., -0.439274,  0.11043 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 14.172975
# Max relative difference: 33865.44
#  x: array([[[1.850944, 1.850944, 1.850944, ..., 1.850944, 1.850944,
#          1.850944],
#         [1.850944, 1.850944, 1.850944, ..., 1.850944, 1.850944,...
#  y: array([[[ 1.850944,  1.850944,  1.850944, ...,  1.781508,  1.73377 ,
#           1.690094],
#         [-1.529978, -1.529978, -1.529978, ..., -1.718444, -1.746722,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 3.735823
# Max relative difference: 85600.3
#  x: array([[[0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,
#          0.110974],
#         [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,...
#  y: array([[[ 0.110974,  0.072982,  0.054712, ..., -0.033717,  0.017383,
#           0.04733 ],
#         [ 0.110974,  0.072982,  0.054712, ..., -0.033717,  0.017383,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131052 / 131072 (100%)
# Max absolute difference: 2.8478417
# Max relative difference: 82122.15
#  x: array([[[0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,
#          0.110974],
#         [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,...
#  y: array([[[ 0.110974,  0.072982,  0.054712, ..., -0.021145,  0.033842,
#           0.008035],
#         [ 0.110974,  0.072982,  0.054712, ..., -0.021145,  0.033842,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131052 / 131072 (100%)
# Max absolute difference: 2.8478417
# Max relative difference: 82122.15
#  x: array([[0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974, 0.110974],
#        [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974, 0.110974],
#        [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974, 0.110974],...
#  y: array([[ 0.110974,  0.072982,  0.054712, ..., -0.021145,  0.033842,
#          0.008035],
#        [ 0.110974,  0.072982,  0.054712, ..., -0.021145,  0.033842,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524240 / 524288 (100%)
# Max absolute difference: 10.970391
# Max relative difference: 42413.695
#  x: array([[0.009638, 0.009638, 0.009638, ..., 0.009638, 0.009638, 0.009638],
#        [0.009638, 0.009638, 0.009638, ..., 0.009638, 0.009638, 0.009638],
#        [0.009638, 0.009638, 0.009638, ..., 0.009638, 0.009638, 0.009638],...
#  y: array([[ 0.009638,  0.004638,  0.007294, ...,  0.07976 ,  0.008571,
#         -0.109329],
#        [ 0.009638,  0.004638,  0.007294, ...,  0.07976 ,  0.008571,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 5.9780025
# Max relative difference: 53087.926
#  x: array([[[0.148719, 0.148719, 0.148719, ..., 0.148719, 0.148719,
#          0.148719],
#         [0.148719, 0.148719, 0.148719, ..., 0.148719, 0.148719,...
#  y: array([[[ 0.148719, -0.257984,  0.546244, ...,  0.820391, -0.099258,
#          -0.123706],
#         [ 0.148719, -0.257984,  0.546244, ...,  0.820391, -0.099258,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131055 / 131072 (100%)
# Max absolute difference: 20.113968
# Max relative difference: 30362.936
#  x: array([[[-0.693233, -0.693233, -0.693233, ..., -0.693233, -0.693233,
#          -0.693233],
#         [-0.693233, -0.693233, -0.693233, ..., -0.693233, -0.693233,...
#  y: array([[[-0.693233, -0.693233, -0.693233, ..., -0.821024, -0.843529,
#          -0.86543 ],
#         [-0.00654 , -0.00654 , -0.00654 , ...,  0.029053,  0.021655,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 2.723987
# Max relative difference: 29287.836
#  x: array([[[0.053245, 0.053245, 0.053245, ..., 0.053245, 0.053245,
#          0.053245],
#         [0.053245, 0.053245, 0.053245, ..., 0.053245, 0.053245,...
#  y: array([[[ 0.053245, -0.118225,  0.061573, ..., -0.028441,  0.014695,
#           0.018549],
#         [ 0.053245, -0.118225,  0.061573, ..., -0.028441,  0.014695,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131047 / 131072 (100%)
# Max absolute difference: 1.9649454
# Max relative difference: 24068.516
#  x: array([[[0.053245, 0.053245, 0.053245, ..., 0.053245, 0.053245,
#          0.053245],
#         [0.053245, 0.053245, 0.053245, ..., 0.053245, 0.053245,...
#  y: array([[[ 0.053245, -0.118225,  0.061573, ...,  0.021296, -0.01788 ,
#           0.012256],
#         [ 0.053245, -0.118225,  0.061573, ...,  0.021296, -0.01788 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131047 / 131072 (100%)
# Max absolute difference: 1.9649454
# Max relative difference: 24068.516
#  x: array([[0.053245, 0.053245, 0.053245, ..., 0.053245, 0.053245, 0.053245],
#        [0.053245, 0.053245, 0.053245, ..., 0.053245, 0.053245, 0.053245],
#        [0.053245, 0.053245, 0.053245, ..., 0.053245, 0.053245, 0.053245],...
#  y: array([[ 0.053245, -0.118225,  0.061573, ...,  0.021296, -0.01788 ,
#          0.012256],
#        [ 0.053245, -0.118225,  0.061573, ...,  0.021296, -0.01788 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524227 / 524288 (100%)
# Max absolute difference: 16.669317
# Max relative difference: 168885.95
#  x: array([[-0.072737, -0.072737, -0.072737, ..., -0.072737, -0.072737,
#         -0.072737],
#        [-0.072737, -0.072737, -0.072737, ..., -0.072737, -0.072737,...
#  y: array([[-0.072737, -0.047771, -0.016222, ..., -0.068065,  0.027962,
#          0.005126],
#        [-0.072737, -0.047771, -0.016222, ..., -0.068066,  0.027962,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 6.7561264
# Max relative difference: 65899.234
#  x: array([[[-0.217756, -0.217756, -0.217756, ..., -0.217756, -0.217756,
#          -0.217756],
#         [-0.217756, -0.217756, -0.217756, ..., -0.217756, -0.217756,...
#  y: array([[[-2.177563e-01, -4.057926e-02,  7.942881e-02, ...,
#          -1.222027e-01, -2.567437e-01, -2.396974e-01],
#         [-2.177563e-01, -4.057935e-02,  7.942884e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 33.153633
# Max relative difference: 342956.25
#  x: array([[[1.696663, 1.696663, 1.696663, ..., 1.696663, 1.696663,
#          1.696663],
#         [1.696663, 1.696663, 1.696663, ..., 1.696663, 1.696663,...
#  y: array([[[ 1.696663e+00,  1.696663e+00,  1.696663e+00, ...,
#           1.674493e+00,  1.666931e+00,  1.657276e+00],
#         [ 1.255733e-01,  1.255734e-01,  1.255733e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 4.107125
# Max relative difference: 15405.046
#  x: array([[[-0.083609, -0.083609, -0.083609, ..., -0.083609, -0.083609,
#          -0.083609],
#         [-0.083609, -0.083609, -0.083609, ..., -0.083609, -0.083609,...
#  y: array([[[-8.360928e-02,  9.421398e-02, -5.866169e-02, ...,
#          -1.151931e-02, -2.844097e-02,  2.532422e-02],
#         [-8.360925e-02,  9.421399e-02, -5.866168e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 2.0626624
# Max relative difference: 43727.047
#  x: array([[[-0.083609, -0.083609, -0.083609, ..., -0.083609, -0.083609,
#          -0.083609],
#         [-0.083609, -0.083609, -0.083609, ..., -0.083609, -0.083609,...
#  y: array([[[-0.083609,  0.094214, -0.058662, ..., -0.01744 ,  0.032394,
#          -0.015279],
#         [-0.083609,  0.094214, -0.058662, ..., -0.01744 ,  0.032394,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 2.0626624
# Max relative difference: 43727.047
#  x: array([[-0.083609, -0.083609, -0.083609, ..., -0.083609, -0.083609,
#         -0.083609],
#        [-0.083609, -0.083609, -0.083609, ..., -0.083609, -0.083609,...
#  y: array([[-0.083609,  0.094214, -0.058662, ..., -0.01744 ,  0.032394,
#         -0.015279],
#        [-0.083609,  0.094214, -0.058662, ..., -0.01744 ,  0.032394,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524059 / 524288 (100%)
# Max absolute difference: 15.968245
# Max relative difference: 7785.5884
#  x: array([[-0.000553, -0.000553, -0.000553, ..., -0.000553, -0.000553,
#         -0.000553],
#        [-0.000553, -0.000553, -0.000553, ..., -0.000553, -0.000553,...
#  y: array([[-0.000553, -0.000949,  0.008609, ...,  0.002296,  0.025531,
#          0.007334],
#        [-0.000553, -0.000949,  0.008609, ...,  0.002296,  0.025531,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 8.647617
# Max relative difference: 10936.954
#  x: array([[[0.551634, 0.551634, 0.551634, ..., 0.551634, 0.551634,
#          0.551634],
#         [0.551634, 0.551634, 0.551634, ..., 0.551634, 0.551634,...
#  y: array([[[ 0.551634, -0.011303,  0.313576, ...,  0.216216,  0.278321,
#           0.084936],
#         [ 0.551634, -0.011303,  0.313576, ...,  0.216216,  0.278321,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131061 / 131072 (100%)
# Max absolute difference: 38.83859
# Max relative difference: 122453.15
#  x: array([[[0.149626, 0.149626, 0.149626, ..., 0.149626, 0.149626,
#          0.149626],
#         [0.149626, 0.149626, 0.149626, ..., 0.149626, 0.149626,...
#  y: array([[[ 1.496261e-01,  1.496261e-01,  1.496261e-01, ...,
#           1.541792e-01,  1.538920e-01,  1.540832e-01],
#         [ 1.110685e+00,  1.110685e+00,  1.110685e+00, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 6.107804
# Max relative difference: 137926.75
#  x: array([[[0.111885, 0.111885, 0.111885, ..., 0.111885, 0.111885,
#          0.111885],
#         [0.111885, 0.111885, 0.111885, ..., 0.111885, 0.111885,...
#  y: array([[[ 1.118845e-01, -7.329548e-02,  8.834306e-03, ...,
#          -1.603155e-01, -4.373023e-02,  5.527420e-02],
#         [ 1.118845e-01, -7.329549e-02,  8.834286e-03, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 3.8389235
# Max relative difference: 113355.76
#  x: array([[[0.111885, 0.111885, 0.111885, ..., 0.111885, 0.111885,
#          0.111885],
#         [0.111885, 0.111885, 0.111885, ..., 0.111885, 0.111885,...
#  y: array([[[ 0.111885, -0.073295,  0.008834, ..., -0.055052, -0.004377,
#           0.02647 ],
#         [ 0.111885, -0.073295,  0.008834, ..., -0.055052, -0.004377,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 3.8389235
# Max relative difference: 113355.76
#  x: array([[0.111885, 0.111885, 0.111885, ..., 0.111885, 0.111885, 0.111885],
#        [0.111885, 0.111885, 0.111885, ..., 0.111885, 0.111885, 0.111885],
#        [0.111885, 0.111885, 0.111885, ..., 0.111885, 0.111885, 0.111885],...
#  y: array([[ 0.111885, -0.073295,  0.008834, ..., -0.055052, -0.004377,
#          0.02647 ],
#        [ 0.111885, -0.073295,  0.008834, ..., -0.055052, -0.004377,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524267 / 524288 (100%)
# Max absolute difference: 3.3770266
# Max relative difference: 98099744.
#  x: array([[-0.045681, -0.045681, -0.045681, ..., -0.045681, -0.045681,
#         -0.045681],
#        [-0.045681, -0.045681, -0.045681, ..., -0.045681, -0.045681,...
#  y: array([[-0.045681,  0.105061, -0.001195, ...,  0.055596,  0.141513,
#          0.002498],
#        [-0.045681,  0.105061, -0.001195, ...,  0.055596,  0.141513,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 6.769949
# Max relative difference: 112634.836
#  x: array([[[0.785479, 0.785479, 0.785479, ..., 0.785479, 0.785479,
#          0.785479],
#         [0.785479, 0.785479, 0.785479, ..., 0.785479, 0.785479,...
#  y: array([[[ 0.785479,  0.566895,  0.094853, ..., -0.790271, -0.110797,
#           0.242074],
#         [ 0.785479,  0.566895,  0.094853, ..., -0.790271, -0.110797,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 25.19277
# Max relative difference: 77572.47
#  x: array([[[1.951457, 1.951457, 1.951457, ..., 1.951457, 1.951457,
#          1.951457],
#         [1.951457, 1.951457, 1.951457, ..., 1.951457, 1.951457,...
#  y: array([[[  1.951457,   1.951456,   1.951456, ...,   1.942088,
#            1.942056,   1.941047],
#         [ -2.768646,  -2.768646,  -2.768646, ...,  -2.764833,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 3.9900832
# Max relative difference: 65320.844
#  x: array([[[-0.027376, -0.027376, -0.027376, ..., -0.027376, -0.027376,
#          -0.027376],
#         [-0.027376, -0.027376, -0.027376, ..., -0.027376, -0.027376,...
#  y: array([[[-2.737607e-02, -1.333458e-02, -8.904648e-02, ...,
#          -5.439343e-02, -5.469035e-02,  5.922398e-03],
#         [-2.737607e-02, -1.333459e-02, -8.904649e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131047 / 131072 (100%)
# Max absolute difference: 2.0595272
# Max relative difference: 38176.918
#  x: array([[[-0.027376, -0.027376, -0.027376, ..., -0.027376, -0.027376,
#          -0.027376],
#         [-0.027376, -0.027376, -0.027376, ..., -0.027376, -0.027376,...
#  y: array([[[-0.027376, -0.013335, -0.089046, ..., -0.046064,  0.021066,
#          -0.0208  ],
#         [-0.027376, -0.013335, -0.089046, ..., -0.046064,  0.021066,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131047 / 131072 (100%)
# Max absolute difference: 2.0595272
# Max relative difference: 38176.918
#  x: array([[-0.027376, -0.027376, -0.027376, ..., -0.027376, -0.027376,
#         -0.027376],
#        [-0.027376, -0.027376, -0.027376, ..., -0.027376, -0.027376,...
#  y: array([[-0.027376, -0.013335, -0.089046, ..., -0.046064,  0.021066,
#         -0.0208  ],
#        [-0.027376, -0.013335, -0.089046, ..., -0.046064,  0.021066,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 523981 / 524288 (99.9%)
# Max absolute difference: 2.4510968
# Max relative difference: 209218.81
#  x: array([[0.002436, 0.002436, 0.002436, ..., 0.002436, 0.002436, 0.002436],
#        [0.002436, 0.002436, 0.002436, ..., 0.002436, 0.002436, 0.002436],
#        [0.002436, 0.002436, 0.002436, ..., 0.002436, 0.002436, 0.002436],...
#  y: array([[ 0.002436, -0.000921,  0.000724, ..., -0.001357,  0.020441,
#         -0.000219],
#        [ 0.002436, -0.000921,  0.000724, ..., -0.001357,  0.020441,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 57.081585
# Max relative difference: 26348.912
#  x: array([[[-0.650415, -0.650415, -0.650415, ..., -0.650415, -0.650415,
#          -0.650415],
#         [-0.650415, -0.650415, -0.650415, ..., -0.650415, -0.650415,...
#  y: array([[[-0.650415,  0.086131,  0.126946, ...,  0.825537,  2.374846,
#           0.803188],
#         [-0.650415,  0.086131,  0.126946, ...,  0.825537,  2.374846,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131062 / 131072 (100%)
# Max absolute difference: 61.315422
# Max relative difference: 3010.1152
#  x: array([[[-0.032485, -0.032485, -0.032485, ..., -0.032485, -0.032485,
#          -0.032485],
#         [-0.032485, -0.032485, -0.032485, ..., -0.032485, -0.032485,...
#  y: array([[[-3.248524e-02, -3.248525e-02, -3.248527e-02, ...,
#           6.755333e-02,  7.860708e-02,  8.690792e-02],
#         [ 4.220543e-02,  4.220544e-02,  4.220545e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 3.59705
# Max relative difference: 3901.3887
#  x: array([[[-0.013749, -0.013749, -0.013749, ..., -0.013749, -0.013749,
#          -0.013749],
#         [-0.013749, -0.013749, -0.013749, ..., -0.013749, -0.013749,...
#  y: array([[[-0.013749,  0.092493,  0.03702 , ..., -0.011387,  0.083624,
#          -0.032308],
#         [-0.013749,  0.092493,  0.03702 , ..., -0.011387,  0.083624,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131051 / 131072 (100%)
# Max absolute difference: 2.1524496
# Max relative difference: 17720.477
#  x: array([[[-0.013749, -0.013749, -0.013749, ..., -0.013749, -0.013749,
#          -0.013749],
#         [-0.013749, -0.013749, -0.013749, ..., -0.013749, -0.013749,...
#  y: array([[[-0.013749,  0.092493,  0.03702 , ..., -0.02284 , -0.002371,
#           0.033615],
#         [-0.013749,  0.092493,  0.03702 , ..., -0.02284 , -0.002371,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131051 / 131072 (100%)
# Max absolute difference: 2.1524496
# Max relative difference: 17720.477
#  x: array([[-0.013749, -0.013749, -0.013749, ..., -0.013749, -0.013749,
#         -0.013749],
#        [-0.013749, -0.013749, -0.013749, ..., -0.013749, -0.013749,...
#  y: array([[-0.013749,  0.092493,  0.03702 , ..., -0.02284 , -0.002371,
#          0.033615],
#        [-0.013749,  0.092493,  0.03702 , ..., -0.02284 , -0.002371,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524233 / 524288 (100%)
# Max absolute difference: 4.6296144
# Max relative difference: 299291.4
#  x: array([[-0.047882, -0.047882, -0.047882, ..., -0.047882, -0.047882,
#         -0.047882],
#        [-0.047882, -0.047882, -0.047882, ..., -0.047882, -0.047882,...
#  y: array([[-4.788162e-02, -1.054291e-04, -6.604420e-02, ...,  2.345947e-02,
#          1.264636e-01, -3.425578e-02],
#        [-4.788162e-02, -1.056623e-04, -6.604420e-02, ...,  2.345949e-02,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 10.436062
# Max relative difference: 55188.76
#  x: array([[[1.194113, 1.194113, 1.194113, ..., 1.194113, 1.194113,
#          1.194113],
#         [1.194113, 1.194113, 1.194113, ..., 1.194113, 1.194113,...
#  y: array([[[ 1.194113e+00, -1.323845e+00,  1.823112e-01, ...,
#          -1.474089e-02,  2.107459e+00,  7.143617e-01],
#         [ 1.194113e+00, -1.323845e+00,  1.823113e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 41.178417
# Max relative difference: 11571.685
#  x: array([[[-0.101112, -0.101112, -0.101112, ..., -0.101112, -0.101112,
#          -0.101112],
#         [-0.101112, -0.101112, -0.101112, ..., -0.101112, -0.101112,...
#  y: array([[[-1.011116e-01, -1.011116e-01, -1.011116e-01, ...,
#          -7.375731e-02, -7.657602e-02, -7.942502e-02],
#         [-1.046206e+00, -1.046206e+00, -1.046206e+00, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131055 / 131072 (100%)
# Max absolute difference: 3.201464
# Max relative difference: 7985.4106
#  x: array([[[-0.055052, -0.055052, -0.055052, ..., -0.055052, -0.055052,
#          -0.055052],
#         [-0.055052, -0.055052, -0.055052, ..., -0.055052, -0.055052,...
#  y: array([[[-5.505180e-02,  1.973702e-02,  4.296672e-02, ...,
#          -1.206592e-02,  4.279196e-02, -4.690741e-02],
#         [-5.505182e-02,  1.973706e-02,  4.296672e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131050 / 131072 (100%)
# Max absolute difference: 2.0350595
# Max relative difference: 137636.9
#  x: array([[[-0.055052, -0.055052, -0.055052, ..., -0.055052, -0.055052,
#          -0.055052],
#         [-0.055052, -0.055052, -0.055052, ..., -0.055052, -0.055052,...
#  y: array([[[-0.055052,  0.019737,  0.042967, ...,  0.016211,  0.00771 ,
#          -0.019585],
#         [-0.055052,  0.019737,  0.042967, ...,  0.016211,  0.00771 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131050 / 131072 (100%)
# Max absolute difference: 2.0350595
# Max relative difference: 137636.9
#  x: array([[-0.055052, -0.055052, -0.055052, ..., -0.055052, -0.055052,
#         -0.055052],
#        [-0.055052, -0.055052, -0.055052, ..., -0.055052, -0.055052,...
#  y: array([[-0.055052,  0.019737,  0.042967, ...,  0.016211,  0.00771 ,
#         -0.019585],
#        [-0.055052,  0.019737,  0.042967, ...,  0.016211,  0.00771 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524250 / 524288 (100%)
# Max absolute difference: 3.412838
# Max relative difference: 3872777.
#  x: array([[0.112713, 0.112713, 0.112713, ..., 0.112713, 0.112713, 0.112713],
#        [0.112713, 0.112713, 0.112713, ..., 0.112713, 0.112713, 0.112713],
#        [0.112713, 0.112713, 0.112713, ..., 0.112713, 0.112713, 0.112713],...
#  y: array([[0.112713, 0.257547, 0.049961, ..., 0.114004, 0.048363, 0.033872],
#        [0.112713, 0.257547, 0.049961, ..., 0.114004, 0.048363, 0.033872],
#        [0.112713, 0.257547, 0.049961, ..., 0.114004, 0.048363, 0.033872],...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 27.495022
# Max relative difference: 9462.784
#  x: array([[[-0.290962, -0.290962, -0.290962, ..., -0.290962, -0.290962,
#          -0.290962],
#         [-0.290962, -0.290962, -0.290962, ..., -0.290962, -0.290962,...
#  y: array([[[-0.290962, -1.408466, -1.050258, ...,  0.3864  ,  0.297145,
#           0.207432],
#         [-0.290963, -1.408466, -1.050259, ...,  0.3864  ,  0.297145,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 56.95332
# Max relative difference: 551846.94
#  x: array([[[2.154472, 2.154472, 2.154472, ..., 2.154472, 2.154472,
#          2.154472],
#         [2.154472, 2.154472, 2.154472, ..., 2.154472, 2.154472,...
#  y: array([[[ 2.154472e+00,  2.154472e+00,  2.154472e+00, ...,
#           2.215172e+00,  2.229574e+00,  2.242239e+00],
#         [-1.863127e+00, -1.863128e+00, -1.863128e+00, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 3.5423021
# Max relative difference: 42551.297
#  x: array([[[0.082995, 0.082995, 0.082995, ..., 0.082995, 0.082995,
#          0.082995],
#         [0.082995, 0.082995, 0.082995, ..., 0.082995, 0.082995,...
#  y: array([[[ 8.299495e-02, -2.864961e-02,  1.672013e-02, ...,
#           7.209902e-02, -3.735290e-02, -3.144727e-02],
#         [ 8.299495e-02, -2.864960e-02,  1.672012e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 2.4038794
# Max relative difference: 422896.3
#  x: array([[[0.082995, 0.082995, 0.082995, ..., 0.082995, 0.082995,
#          0.082995],
#         [0.082995, 0.082995, 0.082995, ..., 0.082995, 0.082995,...
#  y: array([[[ 0.082995, -0.02865 ,  0.01672 , ...,  0.029375, -0.042404,
#           0.045522],
#         [ 0.082995, -0.02865 ,  0.01672 , ...,  0.029375, -0.042404,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 2.4038794
# Max relative difference: 422896.3
#  x: array([[0.082995, 0.082995, 0.082995, ..., 0.082995, 0.082995, 0.082995],
#        [0.082995, 0.082995, 0.082995, ..., 0.082995, 0.082995, 0.082995],
#        [0.082995, 0.082995, 0.082995, ..., 0.082995, 0.082995, 0.082995],...
#  y: array([[ 0.082995, -0.02865 ,  0.01672 , ...,  0.029375, -0.042404,
#          0.045522],
#        [ 0.082995, -0.02865 ,  0.01672 , ...,  0.029375, -0.042404,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524221 / 524288 (100%)
# Max absolute difference: 3.5867841
# Max relative difference: 175552.48
#  x: array([[-0.050744, -0.050744, -0.050744, ..., -0.050744, -0.050744,
#         -0.050744],
#        [-0.050744, -0.050744, -0.050744, ..., -0.050744, -0.050744,...
#  y: array([[-0.050744,  0.076899,  0.022136, ...,  0.164765,  0.100001,
#          0.052142],
#        [-0.050744,  0.076899,  0.022136, ...,  0.164765,  0.100001,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 7.383063
# Max relative difference: 89608.09
#  x: array([[[0.142371, 0.142371, 0.142371, ..., 0.142371, 0.142371,
#          0.142371],
#         [0.142371, 0.142371, 0.142371, ..., 0.142371, 0.142371,...
#  y: array([[[ 0.142371, -0.788295, -0.246142, ...,  0.62091 , -0.201329,
#           0.151155],
#         [ 0.142371, -0.788295, -0.246142, ...,  0.62091 , -0.201329,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131061 / 131072 (100%)
# Max absolute difference: 39.332306
# Max relative difference: 6962.6475
#  x: array([[[0.326459, 0.326459, 0.326459, ..., 0.326459, 0.326459,
#          0.326459],
#         [0.326459, 0.326459, 0.326459, ..., 0.326459, 0.326459,...
#  y: array([[[ 0.326459,  0.326459,  0.326459, ...,  0.311011,  0.313711,
#           0.31643 ],
#         [ 0.990881,  0.990881,  0.990881, ...,  0.929913,  0.930987,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 3.9150107
# Max relative difference: 60867.98
#  x: array([[[-0.047355, -0.047355, -0.047355, ..., -0.047355, -0.047355,
#          -0.047355],
#         [-0.047355, -0.047355, -0.047355, ..., -0.047355, -0.047355,...
#  y: array([[[-0.047355, -0.003235,  0.027647, ...,  0.003778,  0.018768,
#          -0.068589],
#         [-0.047355, -0.003235,  0.027647, ...,  0.003778,  0.018768,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131051 / 131072 (100%)
# Max absolute difference: 2.0854201
# Max relative difference: 385758.22
#  x: array([[[-0.047355, -0.047355, -0.047355, ..., -0.047355, -0.047355,
#          -0.047355],
#         [-0.047355, -0.047355, -0.047355, ..., -0.047355, -0.047355,...
#  y: array([[[-0.047355, -0.003235,  0.027647, ..., -0.009023,  0.010554,
#           0.025377],
#         [-0.047355, -0.003235,  0.027647, ..., -0.009023,  0.010554,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131051 / 131072 (100%)
# Max absolute difference: 2.0854201
# Max relative difference: 385758.22
#  x: array([[-0.047355, -0.047355, -0.047355, ..., -0.047355, -0.047355,
#         -0.047355],
#        [-0.047355, -0.047355, -0.047355, ..., -0.047355, -0.047355,...
#  y: array([[-0.047355, -0.003235,  0.027647, ..., -0.009023,  0.010554,
#          0.025377],
#        [-0.047355, -0.003235,  0.027647, ..., -0.009023,  0.010554,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524254 / 524288 (100%)
# Max absolute difference: 3.8912296
# Max relative difference: 149333.
#  x: array([[0.127425, 0.127425, 0.127425, ..., 0.127425, 0.127425, 0.127425],
#        [0.127425, 0.127425, 0.127425, ..., 0.127425, 0.127425, 0.127425],
#        [0.127425, 0.127425, 0.127425, ..., 0.127425, 0.127425, 0.127425],...
#  y: array([[ 0.127425,  0.05655 , -0.039299, ..., -0.086052,  0.069227,
#         -0.059224],
#        [ 0.127425,  0.056551, -0.039299, ..., -0.086052,  0.069227,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131052 / 131072 (100%)
# Max absolute difference: 6.9869328
# Max relative difference: 47927.438
#  x: array([[[0.550796, 0.550796, 0.550796, ..., 0.550796, 0.550796,
#          0.550796],
#         [0.550796, 0.550796, 0.550796, ..., 0.550796, 0.550796,...
#  y: array([[[ 0.550796, -0.432225, -0.055261, ...,  0.955556, -0.80035 ,
#           0.098826],
#         [ 0.550796, -0.432225, -0.055261, ...,  0.955556, -0.80035 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 18.243288
# Max relative difference: 239006.45
#  x: array([[[3.023679, 3.023679, 3.023679, ..., 3.023679, 3.023679,
#          3.023679],
#         [3.023679, 3.023679, 3.023679, ..., 3.023679, 3.023679,...
#  y: array([[[ 3.023679e+00,  3.023678e+00,  3.023678e+00, ...,
#           3.029484e+00,  3.037781e+00,  3.044928e+00],
#         [ 9.487306e-01,  9.487306e-01,  9.487306e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131055 / 131072 (100%)
# Max absolute difference: 2.8845377
# Max relative difference: 45411.703
#  x: array([[[0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,
#          0.110974],
#         [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,...
#  y: array([[[ 1.109742e-01, -4.701091e-02, -1.824300e-02, ...,
#           1.269488e-02,  8.254809e-02,  1.865234e-01],
#         [ 1.109743e-01, -4.701090e-02, -1.824300e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 1.5334706
# Max relative difference: 62326.258
#  x: array([[[0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,
#          0.110974],
#         [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974,...
#  y: array([[[ 0.110974, -0.047011, -0.018243, ..., -0.055696,  0.001576,
#          -0.019964],
#         [ 0.110974, -0.047011, -0.018243, ..., -0.055696,  0.001576,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 1.5334706
# Max relative difference: 62326.258
#  x: array([[0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974, 0.110974],
#        [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974, 0.110974],
#        [0.110974, 0.110974, 0.110974, ..., 0.110974, 0.110974, 0.110974],...
#  y: array([[ 0.110974, -0.047011, -0.018243, ..., -0.055696,  0.001576,
#         -0.019964],
#        [ 0.110974, -0.047011, -0.018243, ..., -0.055696,  0.001576,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524223 / 524288 (100%)
# Max absolute difference: 4.0325212
# Max relative difference: 289271.84
#  x: array([[-0.082875, -0.082875, -0.082875, ..., -0.082875, -0.082875,
#         -0.082875],
#        [-0.082875, -0.082875, -0.082875, ..., -0.082875, -0.082875,...
#  y: array([[-0.082875, -0.029846, -0.058295, ...,  0.154872,  0.191306,
#          0.203534],
#        [-0.082875, -0.029846, -0.058295, ...,  0.154872,  0.191306,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 9.171938
# Max relative difference: 214339.39
#  x: array([[[1.948273, 1.948273, 1.948273, ..., 1.948273, 1.948273,
#          1.948273],
#         [1.948273, 1.948273, 1.948273, ..., 1.948273, 1.948273,...
#  y: array([[[ 1.948273, -0.355274, -0.070722, ..., -0.249204,  1.070692,
#          -1.763864],
#         [ 1.948273, -0.355274, -0.070722, ..., -0.249204,  1.070692,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 31.279318
# Max relative difference: 53701.39
#  x: array([[[2.390944, 2.390944, 2.390944, ..., 2.390944, 2.390944,
#          2.390944],
#         [2.390944, 2.390944, 2.390944, ..., 2.390944, 2.390944,...
#  y: array([[[ 2.390944e+00,  2.390944e+00,  2.390944e+00, ...,
#           2.427362e+00,  2.437556e+00,  2.446170e+00],
#         [-1.401698e+00, -1.401698e+00, -1.401698e+00, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 4.0012207
# Max relative difference: 30559.57
#  x: array([[[-0.018044, -0.018044, -0.018044, ..., -0.018044, -0.018044,
#          -0.018044],
#         [-0.018044, -0.018044, -0.018044, ..., -0.018044, -0.018044,...
#  y: array([[[-0.018044, -0.091411, -0.168368, ...,  0.137008, -0.120064,
#          -0.072629],
#         [-0.018044, -0.091411, -0.168368, ...,  0.137008, -0.120064,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131035 / 131072 (100%)
# Max absolute difference: 2.076216
# Max relative difference: 16802.383
#  x: array([[[-0.018044, -0.018044, -0.018044, ..., -0.018044, -0.018044,
#          -0.018044],
#         [-0.018044, -0.018044, -0.018044, ..., -0.018044, -0.018044,...
#  y: array([[[-0.018044, -0.091411, -0.168368, ...,  0.002302, -0.058753,
#           0.041829],
#         [-0.018044, -0.091411, -0.168368, ...,  0.002302, -0.058753,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131035 / 131072 (100%)
# Max absolute difference: 2.076216
# Max relative difference: 16802.383
#  x: array([[-0.018044, -0.018044, -0.018044, ..., -0.018044, -0.018044,
#         -0.018044],
#        [-0.018044, -0.018044, -0.018044, ..., -0.018044, -0.018044,...
#  y: array([[-0.018044, -0.091411, -0.168368, ...,  0.002302, -0.058753,
#          0.041829],
#        [-0.018044, -0.091411, -0.168368, ...,  0.002302, -0.058753,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524246 / 524288 (100%)
# Max absolute difference: 2.0865788
# Max relative difference: 256179.78
#  x: array([[0.110568, 0.110568, 0.110568, ..., 0.110568, 0.110568, 0.110568],
#        [0.110568, 0.110568, 0.110568, ..., 0.110568, 0.110568, 0.110568],
#        [0.110568, 0.110568, 0.110568, ..., 0.110568, 0.110568, 0.110568],...
#  y: array([[ 0.110568, -0.044877, -0.061378, ..., -0.030172,  0.125128,
#         -0.029323],
#        [ 0.110568, -0.044877, -0.061378, ..., -0.030172,  0.125128,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 8.139444
# Max relative difference: 421085.3
#  x: array([[[0.544325, 0.544325, 0.544325, ..., 0.544325, 0.544325,
#          0.544325],
#         [0.544325, 0.544325, 0.544325, ..., 0.544325, 0.544325,...
#  y: array([[[ 0.544325,  1.29637 ,  0.415751, ...,  1.767701,  0.894819,
#           0.825459],
#         [ 0.544325,  1.29637 ,  0.415751, ...,  1.767701,  0.894819,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 36.675186
# Max relative difference: 7935.9443
#  x: array([[[-0.205737, -0.205737, -0.205737, ..., -0.205737, -0.205737,
#          -0.205737],
#         [-0.205737, -0.205737, -0.205737, ..., -0.205737, -0.205737,...
#  y: array([[[ -0.205737,  -0.205737,  -0.205737, ...,  -0.183041,
#           -0.185854,  -0.186481],
#         [  0.753302,   0.753302,   0.753302, ...,   0.759287,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131046 / 131072 (100%)
# Max absolute difference: 4.379333
# Max relative difference: 4923.099
#  x: array([[[0.017006, 0.017006, 0.017006, ..., 0.017006, 0.017006,
#          0.017006],
#         [0.017006, 0.017006, 0.017006, ..., 0.017006, 0.017006,...
#  y: array([[[ 0.017006,  0.027664, -0.036807, ..., -0.029356, -0.044565,
#          -0.00179 ],
#         [ 0.017006,  0.027664, -0.036807, ..., -0.029356, -0.044565,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131034 / 131072 (100%)
# Max absolute difference: 2.1726916
# Max relative difference: 35615.67
#  x: array([[[0.017006, 0.017006, 0.017006, ..., 0.017006, 0.017006,
#          0.017006],
#         [0.017006, 0.017006, 0.017006, ..., 0.017006, 0.017006,...
#  y: array([[[ 0.017006,  0.027664, -0.036807, ..., -0.000572, -0.024193,
#          -0.020578],
#         [ 0.017006,  0.027664, -0.036807, ..., -0.000572, -0.024193,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131034 / 131072 (100%)
# Max absolute difference: 2.1726916
# Max relative difference: 35615.67
#  x: array([[0.017006, 0.017006, 0.017006, ..., 0.017006, 0.017006, 0.017006],
#        [0.017006, 0.017006, 0.017006, ..., 0.017006, 0.017006, 0.017006],
#        [0.017006, 0.017006, 0.017006, ..., 0.017006, 0.017006, 0.017006],...
#  y: array([[ 0.017006,  0.027664, -0.036807, ..., -0.000572, -0.024193,
#         -0.020578],
#        [ 0.017006,  0.027664, -0.036807, ..., -0.000572, -0.024193,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524207 / 524288 (100%)
# Max absolute difference: 7.5504503
# Max relative difference: 319916.72
#  x: array([[-0.078658, -0.078658, -0.078658, ..., -0.078658, -0.078658,
#         -0.078658],
#        [-0.078658, -0.078658, -0.078658, ..., -0.078658, -0.078658,...
#  y: array([[-0.078658,  0.183004, -0.042838, ..., -0.017829,  0.153535,
#         -0.039136],
#        [-0.078658,  0.183004, -0.042838, ..., -0.017829,  0.153535,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 23.557528
# Max relative difference: 463705.72
#  x: array([[[0.946638, 0.946638, 0.946638, ..., 0.946638, 0.946638,
#          0.946638],
#         [0.946638, 0.946638, 0.946638, ..., 0.946638, 0.946638,...
#  y: array([[[ 0.946638, -4.541362,  0.166896, ..., -0.802669, -1.729379,
#           0.471473],
#         [ 0.946638, -4.541362,  0.166896, ..., -0.802669, -1.729378,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 49.68452
# Max relative difference: 35488.93
#  x: array([[[-0.840858, -0.840858, -0.840858, ..., -0.840858, -0.840858,
#          -0.840858],
#         [-0.840858, -0.840858, -0.840858, ..., -0.840858, -0.840858,...
#  y: array([[[-8.408575e-01, -8.408575e-01, -8.408577e-01, ...,
#          -7.799831e-01, -7.748014e-01, -7.706414e-01],
#         [-9.911448e-01, -9.911447e-01, -9.911448e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131054 / 131072 (100%)
# Max absolute difference: 3.999368
# Max relative difference: 54538.69
#  x: array([[[-0.087264, -0.087264, -0.087264, ..., -0.087264, -0.087264,
#          -0.087264],
#         [-0.087264, -0.087264, -0.087264, ..., -0.087264, -0.087264,...
#  y: array([[[-0.087264,  0.155943,  0.065474, ..., -0.081628,  0.126773,
#          -0.023005],
#         [-0.087264,  0.155943,  0.065474, ..., -0.081628,  0.126773,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 2.6694086
# Max relative difference: 113170.24
#  x: array([[[-0.087264, -0.087264, -0.087264, ..., -0.087264, -0.087264,
#          -0.087264],
#         [-0.087264, -0.087264, -0.087264, ..., -0.087264, -0.087264,...
#  y: array([[[-0.087264,  0.155943,  0.065474, ...,  0.031687, -0.026584,
#           0.004189],
#         [-0.087264,  0.155943,  0.065474, ...,  0.031687, -0.026584,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 2.6694086
# Max relative difference: 113170.24
#  x: array([[-0.087264, -0.087264, -0.087264, ..., -0.087264, -0.087264,
#         -0.087264],
#        [-0.087264, -0.087264, -0.087264, ..., -0.087264, -0.087264,...
#  y: array([[-0.087264,  0.155943,  0.065474, ...,  0.031687, -0.026584,
#          0.004189],
#        [-0.087264,  0.155943,  0.065474, ...,  0.031687, -0.026584,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524272 / 524288 (100%)
# Max absolute difference: 2.2788773
# Max relative difference: 197688.5
#  x: array([[0.14091, 0.14091, 0.14091, ..., 0.14091, 0.14091, 0.14091],
#        [0.14091, 0.14091, 0.14091, ..., 0.14091, 0.14091, 0.14091],
#        [0.14091, 0.14091, 0.14091, ..., 0.14091, 0.14091, 0.14091],...
#  y: array([[ 0.14091 ,  0.061324,  0.048719, ...,  0.056087,  0.00565 ,
#         -0.054153],
#        [ 0.14091 ,  0.061324,  0.048719, ...,  0.056087,  0.00565 ,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 73.16312
# Max relative difference: 39856.684
#  x: array([[[-0.093965, -0.093965, -0.093965, ..., -0.093965, -0.093965,
#          -0.093965],
#         [-0.093965, -0.093965, -0.093965, ..., -0.093965, -0.093965,...
#  y: array([[[-0.093965,  0.289992, -0.246622, ..., -1.317659, -1.001178,
#           0.722653],
#         [-0.093965,  0.289992, -0.246622, ..., -1.317659, -1.001178,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 45.47404
# Max relative difference: 89279.46
#  x: array([[[1.67824, 1.67824, 1.67824, ..., 1.67824, 1.67824, 1.67824],
#         [1.67824, 1.67824, 1.67824, ..., 1.67824, 1.67824, 1.67824],
#         [1.67824, 1.67824, 1.67824, ..., 1.67824, 1.67824, 1.67824],...
#  y: array([[[ 1.678240e+00,  1.678240e+00,  1.678240e+00, ...,
#           1.758191e+00,  1.785264e+00,  1.808502e+00],
#         [-1.747034e-01, -1.747034e-01, -1.747035e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131049 / 131072 (100%)
# Max absolute difference: 3.686863
# Max relative difference: 3026.1328
#  x: array([[[0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],
#         [0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],
#         [0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],...
#  y: array([[[ 6.740091e-03,  1.520122e-02,  3.028878e-01, ...,
#           6.086579e-02, -5.796858e-02, -1.649504e-01],
#         [ 6.740079e-03,  1.520122e-02,  3.028878e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131032 / 131072 (100%)
# Max absolute difference: 1.6647989
# Max relative difference: 12393.647
#  x: array([[[0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],
#         [0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],
#         [0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],...
#  y: array([[[ 0.00674 ,  0.015201,  0.302888, ...,  0.012365,  0.006822,
#           0.001832],
#         [ 0.00674 ,  0.015201,  0.302888, ...,  0.012365,  0.006822,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131032 / 131072 (100%)
# Max absolute difference: 1.6647989
# Max relative difference: 12393.647
#  x: array([[0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],
#        [0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],
#        [0.00674, 0.00674, 0.00674, ..., 0.00674, 0.00674, 0.00674],...
#  y: array([[ 0.00674 ,  0.015201,  0.302888, ...,  0.012365,  0.006822,
#          0.001832],
#        [ 0.00674 ,  0.015201,  0.302888, ...,  0.012365,  0.006822,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524228 / 524288 (100%)
# Max absolute difference: 4.299029
# Max relative difference: 114668.14
#  x: array([[-0.049436, -0.049436, -0.049436, ..., -0.049436, -0.049436,
#         -0.049436],
#        [-0.049436, -0.049436, -0.049436, ..., -0.049436, -0.049436,...
#  y: array([[-0.049436,  0.044037,  0.14876 , ...,  0.06973 , -0.091903,
#         -0.090541],
#        [-0.049436,  0.044037,  0.14876 , ...,  0.06973 , -0.091903,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 23.346483
# Max relative difference: 8161.103
#  x: array([[[0.256872, 0.256872, 0.256872, ..., 0.256872, 0.256872,
#          0.256872],
#         [0.256872, 0.256872, 0.256872, ..., 0.256872, 0.256872,...
#  y: array([[[ 0.256872,  0.241366, -0.726588, ..., -2.607897, -1.265234,
#          -1.321385],
#         [ 0.256872,  0.241366, -0.726588, ..., -2.607897, -1.265235,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 51.89107
# Max relative difference: 12472.306
#  x: array([[[0.57301, 0.57301, 0.57301, ..., 0.57301, 0.57301, 0.57301],
#         [0.57301, 0.57301, 0.57301, ..., 0.57301, 0.57301, 0.57301],
#         [0.57301, 0.57301, 0.57301, ..., 0.57301, 0.57301, 0.57301],...
#  y: array([[[  0.57301 ,   0.57301 ,   0.57301 , ...,   0.643609,
#            0.667821,   0.689978],
#         [ -0.692114,  -0.692114,  -0.692114, ...,  -0.699057,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 3.2421856
# Max relative difference: 639017.44
#  x: array([[[-0.16567, -0.16567, -0.16567, ..., -0.16567, -0.16567,
#          -0.16567],
#         [-0.16567, -0.16567, -0.16567, ..., -0.16567, -0.16567,...
#  y: array([[[-0.16567 ,  0.01952 ,  0.001416, ..., -0.075232, -0.039874,
#          -0.183478],
#         [-0.16567 ,  0.01952 ,  0.001416, ..., -0.075232, -0.039874,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 2.5909107
# Max relative difference: 375629.8
#  x: array([[[-0.16567, -0.16567, -0.16567, ..., -0.16567, -0.16567,
#          -0.16567],
#         [-0.16567, -0.16567, -0.16567, ..., -0.16567, -0.16567,...
#  y: array([[[-0.16567 ,  0.01952 ,  0.001416, ..., -0.073452, -0.042192,
#           0.078358],
#         [-0.16567 ,  0.01952 ,  0.001416, ..., -0.073452, -0.042192,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 2.5909107
# Max relative difference: 375629.8
#  x: array([[-0.16567, -0.16567, -0.16567, ..., -0.16567, -0.16567, -0.16567],
#        [-0.16567, -0.16567, -0.16567, ..., -0.16567, -0.16567, -0.16567],
#        [-0.16567, -0.16567, -0.16567, ..., -0.16567, -0.16567, -0.16567],...
#  y: array([[-0.16567 ,  0.01952 ,  0.001416, ..., -0.073452, -0.042192,
#          0.078358],
#        [-0.16567 ,  0.01952 ,  0.001416, ..., -0.073452, -0.042192,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524253 / 524288 (100%)
# Max absolute difference: 2.7054186
# Max relative difference: 267359.44
#  x: array([[0.052787, 0.052787, 0.052787, ..., 0.052787, 0.052787, 0.052787],
#        [0.052787, 0.052787, 0.052787, ..., 0.052787, 0.052787, 0.052787],
#        [0.052787, 0.052787, 0.052787, ..., 0.052787, 0.052787, 0.052787],...
#  y: array([[ 0.052787, -0.164791,  0.129344, ...,  0.129466, -0.002623,
#         -0.0651  ],
#        [ 0.052787, -0.164791,  0.129344, ...,  0.129466, -0.002623,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131056 / 131072 (100%)
# Max absolute difference: 10.745222
# Max relative difference: 161679.06
#  x: array([[[-1.216656, -1.216656, -1.216656, ..., -1.216656, -1.216656,
#          -1.216656],
#         [-1.216656, -1.216656, -1.216656, ..., -1.216656, -1.216656,...
#  y: array([[[-1.216656,  0.572904, -0.338526, ..., -1.405134, -0.103696,
#          -0.710178],
#         [-1.216656,  0.572904, -0.338526, ..., -1.405134, -0.103696,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 40.810516
# Max relative difference: 28455.986
#  x: array([[[0.524503, 0.524503, 0.524503, ..., 0.524503, 0.524503,
#          0.524503],
#         [0.524503, 0.524503, 0.524503, ..., 0.524503, 0.524503,...
#  y: array([[[ 0.524503,  0.524503,  0.524503, ...,  0.48703 ,  0.49126 ,
#           0.493928],
#         [ 0.758318,  0.758318,  0.758318, ...,  0.660279,  0.663237,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 3.230019
# Max relative difference: 360457.
#  x: array([[[0.322275, 0.322275, 0.322275, ..., 0.322275, 0.322275,
#          0.322275],
#         [0.322275, 0.322275, 0.322275, ..., 0.322275, 0.322275,...
#  y: array([[[ 3.222746e-01,  1.592517e-02, -1.497708e-01, ...,
#           2.410795e-01,  4.821745e-02, -4.610847e-01],
#         [ 3.222746e-01,  1.592520e-02, -1.497708e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 2.0657177
# Max relative difference: 676573.1
#  x: array([[[0.322275, 0.322275, 0.322275, ..., 0.322275, 0.322275,
#          0.322275],
#         [0.322275, 0.322275, 0.322275, ..., 0.322275, 0.322275,...
#  y: array([[[ 0.322275,  0.015925, -0.149771, ...,  0.037069, -0.037685,
#          -0.021145],
#         [ 0.322275,  0.015925, -0.149771, ...,  0.037069, -0.037685,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 2.0657177
# Max relative difference: 676573.1
#  x: array([[0.322275, 0.322275, 0.322275, ..., 0.322275, 0.322275, 0.322275],
#        [0.322275, 0.322275, 0.322275, ..., 0.322275, 0.322275, 0.322275],
#        [0.322275, 0.322275, 0.322275, ..., 0.322275, 0.322275, 0.322275],...
#  y: array([[ 0.322275,  0.015925, -0.149771, ...,  0.037069, -0.037685,
#         -0.021145],
#        [ 0.322275,  0.015925, -0.149771, ...,  0.037069, -0.037685,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524261 / 524288 (100%)
# Max absolute difference: 4.2825227
# Max relative difference: 279786.22
#  x: array([[0.080797, 0.080797, 0.080797, ..., 0.080797, 0.080797, 0.080797],
#        [0.080797, 0.080797, 0.080797, ..., 0.080797, 0.080797, 0.080797],
#        [0.080797, 0.080797, 0.080797, ..., 0.080797, 0.080797, 0.080797],...
#  y: array([[ 0.080797, -0.153762, -0.105972, ...,  0.021226,  0.096737,
#         -0.122422],
#        [ 0.080797, -0.153762, -0.105972, ...,  0.021226,  0.096737,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 17.01115
# Max relative difference: 25754.508
#  x: array([[[0.689228, 0.689228, 0.689228, ..., 0.689228, 0.689228,
#          0.689228],
#         [0.689228, 0.689228, 0.689228, ..., 0.689228, 0.689228,...
#  y: array([[[ 0.689228, -0.501556, -1.667346, ...,  0.932277,  1.234224,
#           0.908113],
#         [ 0.689228, -0.501556, -1.667346, ...,  0.932277,  1.234224,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131055 / 131072 (100%)
# Max absolute difference: 40.554295
# Max relative difference: 254625.53
#  x: array([[[2.549703, 2.549703, 2.549703, ..., 2.549703, 2.549703,
#          2.549703],
#         [2.549703, 2.549703, 2.549703, ..., 2.549703, 2.549703,...
#  y: array([[[ 2.549703e+00,  2.549703e+00,  2.549703e+00, ...,
#           2.611074e+00,  2.643755e+00,  2.670453e+00],
#         [-1.708790e+00, -1.708790e+00, -1.708790e+00, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 8.751081
# Max relative difference: 170434.19
#  x: array([[[1.005702, 1.005702, 1.005702, ..., 1.005702, 1.005702,
#          1.005702],
#         [1.005702, 1.005702, 1.005702, ..., 1.005702, 1.005702,...
#  y: array([[[ 1.005702e+00,  3.487267e-01,  7.793968e-01, ...,
#          -4.827510e-02, -1.287058e-01,  1.008587e+00],
#         [ 1.005702e+00,  3.487266e-01,  7.793968e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 8.751081
# Max relative difference: 4465785.
#  x: array([[[1.005702, 1.005702, 1.005702, ..., 1.005702, 1.005702,
#          1.005702],
#         [1.005702, 1.005702, 1.005702, ..., 1.005702, 1.005702,...
#  y: array([[[ 1.005702e+00,  3.487267e-01,  7.793968e-01, ...,
#          -4.981756e-04,  5.665652e-03,  4.691012e-02],
#         [ 1.005702e+00,  3.487267e-01,  7.793968e-01, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 8.751081
# Max relative difference: 4465785.
#  x: array([[1.005702, 1.005702, 1.005702, ..., 1.005702, 1.005702, 1.005702],
#        [1.005702, 1.005702, 1.005702, ..., 1.005702, 1.005702, 1.005702],
#        [1.005702, 1.005702, 1.005702, ..., 1.005702, 1.005702, 1.005702],...
#  y: array([[ 1.005702e+00,  3.487267e-01,  7.793968e-01, ..., -4.981756e-04,
#          5.665652e-03,  4.691012e-02],
#        [ 1.005702e+00,  3.487267e-01,  7.793968e-01, ..., -4.981924e-04,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524273 / 524288 (100%)
# Max absolute difference: 19.506933
# Max relative difference: 162566.02
#  x: array([[0.070093, 0.070093, 0.070093, ..., 0.070093, 0.070093, 0.070093],
#        [0.070093, 0.070093, 0.070093, ..., 0.070093, 0.070093, 0.070093],
#        [0.070093, 0.070093, 0.070093, ..., 0.070093, 0.070093, 0.070093],...
#  y: array([[ 0.070093,  0.927529,  0.349724, ..., -0.005895, -0.030193,
#         -0.095713],
#        [ 0.070093,  0.927529,  0.349724, ..., -0.005895, -0.030193,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131058 / 131072 (100%)
# Max absolute difference: 103.670296
# Max relative difference: 47469.504
#  x: array([[[0.995929, 0.995929, 0.995929, ..., 0.995929, 0.995929,
#          0.995929],
#         [0.995929, 0.995929, 0.995929, ..., 0.995929, 0.995929,...
#  y: array([[[ 0.995929,  0.080994, -0.581342, ..., -0.292103,  0.206203,
#           0.9273  ],
#         [ 0.995929,  0.080994, -0.581342, ..., -0.292102,  0.206203,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 46.04348
# Max relative difference: 196140.23
#  x: array([[[-0.493389, -0.493389, -0.493389, ..., -0.493389, -0.493389,
#          -0.493389],
#         [-0.493389, -0.493389, -0.493389, ..., -0.493389, -0.493389,...
#  y: array([[[-4.933887e-01, -4.933886e-01, -4.933887e-01, ...,
#          -7.919166e-02,  5.073418e-02,  1.623650e-01],
#         [ 1.913483e+00,  1.913483e+00,  1.913483e+00, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 5.7821646
# Max relative difference: 13027.925
#  x: array([[[-0.092802, -0.092802, -0.092802, ..., -0.092802, -0.092802,
#          -0.092802],
#         [-0.092802, -0.092802, -0.092802, ..., -0.092802, -0.092802,...
#  y: array([[[-9.280184e-02, -1.159546e-01,  2.963499e-02, ...,
#          -1.095048e-01, -1.149143e-01,  2.580456e-01],
#         [-9.280186e-02, -1.159545e-01,  2.963497e-02, ...,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 4.0979905
# Max relative difference: 1441019.6
#  x: array([[[-0.092802, -0.092802, -0.092802, ..., -0.092802, -0.092802,
#          -0.092802],
#         [-0.092802, -0.092802, -0.092802, ..., -0.092802, -0.092802,...
#  y: array([[[-0.092802, -0.115955,  0.029635, ...,  0.119054, -0.064616,
#          -0.037401],
#         [-0.092802, -0.115955,  0.029635, ...,  0.119054, -0.064616,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 4.0979905
# Max relative difference: 1441019.6
#  x: array([[-0.092802, -0.092802, -0.092802, ..., -0.092802, -0.092802,
#         -0.092802],
#        [-0.092802, -0.092802, -0.092802, ..., -0.092802, -0.092802,...
#  y: array([[-0.092802, -0.115955,  0.029635, ...,  0.119054, -0.064616,
#         -0.037401],
#        [-0.092802, -0.115955,  0.029635, ...,  0.119054, -0.064616,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
# Mismatched elements: 524091 / 524288 (100%)
# Max absolute difference: 26.020678
# Max relative difference: 2186.5562
#  x: array([[-0.000631, -0.000631, -0.000631, ..., -0.000631, -0.000631,
#         -0.000631],
#        [-0.000631, -0.000631, -0.000631, ..., -0.000631, -0.000631,...
#  y: array([[-6.314622e-04, -1.515209e-01,  1.032860e+00, ..., -1.614523e-01,
#          5.076567e-01, -1.226874e-01],
#        [-6.317447e-04, -1.515209e-01,  1.032861e+00, ..., -1.614524e-01,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131060 / 131072 (100%)
# Max absolute difference: 9.299807
# Max relative difference: 1445812.1
#  x: array([[[-1.077215, -1.077215, -1.077215, ..., -1.077215, -1.077215,
#          -1.077215],
#         [-1.077215, -1.077215, -1.077215, ..., -1.077215, -1.077215,...
#  y: array([[[-1.077215, -0.737891,  0.565912, ..., -1.050993,  0.254314,
#           0.173379],
#         [-1.077215, -0.737892,  0.565912, ..., -1.050993,  0.254316,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131057 / 131072 (100%)
# Max absolute difference: 32.20655
# Max relative difference: 7520466.
#  x: array([[[-2.017146, -2.017146, -2.017146, ..., -2.017146, -2.017146,
#          -2.017146],
#         [-2.017146, -2.017146, -2.017146, ..., -2.017146, -2.017146,...
#  y: array([[[-2.017146, -2.017145, -2.017145, ..., -1.713519, -1.672338,
#          -1.610281],
#         [-1.07886 , -1.07886 , -1.078859, ..., -0.430205, -0.374982,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 64, 128]), [16, 64, 128], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 2.5848427
# Max relative difference: 9209.909
#  x: array([[[-0.013242, -0.013242, -0.013242, ..., -0.013242, -0.013242,
#          -0.013242],
#         [-0.013242, -0.013242, -0.013242, ..., -0.013242, -0.013242,...
#  y: array([[[-0.013242,  0.050285, -0.015645, ...,  0.223655, -0.029152,
#           0.038309],
#         [-0.013242,  0.050285, -0.015645, ...,  0.223654, -0.029152,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 16, 128, 64]), [16, 128, 64], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 2.3001497
# Max relative difference: 2833.5105
#  x: array([[[-0.013242, -0.013242, -0.013242, ..., -0.013242, -0.013242,
#          -0.013242],
#         [-0.013242, -0.013242, -0.013242, ..., -0.013242, -0.013242,...
#  y: array([[[-0.013242,  0.050285, -0.015645, ...,  0.380386,  0.052017,
#           0.580691],
#         [-0.013242,  0.050285, -0.015645, ...,  0.380386,  0.052016,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 16, 64]), [1, 128, 1024], [0, 0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 131059 / 131072 (100%)
# Max absolute difference: 2.3001497
# Max relative difference: 2833.5105
#  x: array([[-0.013242, -0.013242, -0.013242, ..., -0.013242, -0.013242,
#         -0.013242],
#        [-0.013242, -0.013242, -0.013242, ..., -0.013242, -0.013242,...
#  y: array([[-0.013242,  0.050285, -0.015645, ...,  0.380386,  0.052017,
#          0.580691],
#        [-0.013242,  0.050285, -0.015645, ...,  0.380386,  0.052016,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 1024]), [128, 1024], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 524145 / 524288 (100%)
# Max absolute difference: 29.939627
# Max relative difference: 571538.06
#  x: array([[-0.136265, -0.136265, -0.136265, ..., -0.136265, -0.136265,
#         -0.136265],
#        [-0.136265, -0.136265, -0.136265, ..., -0.136265, -0.136265,...
#  y: array([[-0.136265, -0.023468, -0.122506, ..., -0.161708, -0.167916,
#         -0.079876],
#        [-0.136265, -0.023469, -0.122506, ..., -0.161708, -0.167916,...*; Dispatched function name: *aten._reshape_alias.default*; Dispatched function args: *[torch.Size([1, 128, 4096]), [128, 4096], [0, 0]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# Target triple found:x86_64-linux-gnu
# /home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py:173: UserWarning: Lockstep accuracy verification failed with error: *
# Not equal to tolerance rtol=0.0001, atol=1e-05
#
# Mismatched elements: 1 / 1 (100%)
# Max absolute difference: 128
# Max relative difference: inf
#  x: array([128])
#  y: array([False])*; Dispatched function name: *aten.sum.dim_IntList*; Dispatched function args: *[torch.Size([1, 128]), [-1]]*; Dispatched function kwargs: *[]*;
#   warnings.warn(
# Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 126, in __torch_dispatch__
#     eager_module = build_mlir_module(func, normalized_kwargs)
#   File "/home/chi/src/ubuntu20/shark/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir/torch_mlir/eager_mode/ir_building.py", line 343, in build_mlir_module
#     assert len(annotations) == len(
# AssertionError: Number of annotations and number of graph inputs differs.
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "/home/chi/src/ubuntu20/shark/SHARK/tank/bloom_model.py", line 37, in <module>
#     output = model(eager_input_batch) # RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1190, in _call_impl
#     return forward_call(*input, **kwargs)
#   File "/home/chi/src/ubuntu20/shark/SHARK/tank/bloom_model.py", line 23, in forward
#     return self.model.forward(tokens)[0]
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/transformers/models/bloom/modeling_bloom.py", line 955, in forward
#     sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/_tensor.py", line 1270, in __torch_function__
#     ret = func(*args, **kwargs)
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark/torch_mlir_lockstep_tensor.py", line 194, in __torch_dispatch__
#     out = func(*unwrapped_args, **unwrapped_kwargs)
#   File "/home/chi/src/ubuntu20/shark/SHARK/shark.venv/lib/python3.10/site-packages/torch/_ops.py", line 60, in __call__
#     return self._op(*args, **kwargs or {})
# RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.