import torch
import torch_mlir
from transformers import AutoModelForSequenceClassification

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


############# Op TRACE LOG 31-43copy from squeezenet_lockstep.py#50 #############

# from shark.torch_mlir_lockstep_tensor import TorchMLIRLockstepTensor
# input_detached_clone = test_input.clone()
# eager_input_batch = TorchMLIRLockstepTensor(input_detached_clone)
# print("getting torch-mlir result")
# output = model(eager_input_batch) # RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
# print("output: ", output)
# static_output = output.elem
# print("static_output: ", static_output)
# shark_out = static_output[0]
# print("The obtained result via shark is: ", shark_out)
# print("The golden result is:", actual_out)


from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
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

# QUANTIZATION SETUP
model = fx_g

# test1: static quantization  SUCCESS
# golden_out VS shark_out
# tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[  7.2043, -17.0265]])
model.eval()
model.qconfig = torch.quantization.qconfig.float16_static_qconfig

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
model_fp32_prepared(test_input)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_fp16 = torch.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in fp16
res = model_fp16(test_input)
print("res: ", res)
fx_g = model_fp16



# # test2: dynamic quantization  SUCCESS
# # golden_out VS shark_out
# # tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[  7.2043, -17.0265]])
# model.eval()
# model_fp16 = torch.quantization.quantize_dynamic(
#     model,  # the original model
#     {torch.nn.Linear},  # a set of layers to dynamically quantize
#     dtype=torch.float16)  # the target dtype for quantized weights
# # res:  tensor([[  7.2050, -17.0270]])

# # run the model, relevant calculations will happen in fp16
# res = model_fp16(test_input)
# print("res: ", res)
# fx_g = model_fp16



# # test3:quantize_fx SUCCESS, But shark_out vary in different run
# # golden_out VS shark_out
# # tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[-5.0346, -3.0476]])
# # tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[-8.9322, -2.2289]])
# # tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[-7.9967, -6.8403]])
# import torch.quantization.quantize_fx as quantize_fx
# # import copy
# #
# # model_to_quantize = copy.deepcopy(model)
# qconfig_dict = {"": torch.quantization.qconfig.float16_static_qconfig}
# model.eval()
# # prepare
# model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=((1, 128),))
# # calibrate (not shown)
# # quantize
# model_fp16 = quantize_fx.convert_fx(model_prepared)

# # run the model, relevant calculations will happen in fp16
# res = model_fp16(test_input)
# print("res: ", res)
# fx_g = model_fp16



ts_g = torch.jit.script(fx_g)
# print(ts_g.graph)

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
# QUANTIZATION 
# test1: static quantization:  
#       tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[  7.2043, -17.0265]])
# test2: dynamic quantization:
#       tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[  7.2043, -17.0265]])
# test3:quantize_fx : 
#       tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[-5.0346, -3.0476]])
#       tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[-8.9322, -2.2289]])
#       tensor([[  7.2041, -17.0263]], grad_fn=<IndexBackward0>) tensor([[-7.9967, -6.8403]])
