"""
A simple example for bounding neural network outputs under input perturbations.

This example serves as a skeleton for robustness verification of neural networks.
"""
import sys
import os
import gc


DISABLE_TF32 = os.environ.get("CF_DISABLE_TF32", "0") not in ("", "0")
if DISABLE_TF32:
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import torch
if DISABLE_TF32:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
import torchvision
from math import pi
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
from collections import deque

# auto_lirpa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', 'auto_lirpa'))
# sys.path.append(auto_lirpa_path)

# print("sys.path:", sys.path)
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from torch.nn import Module
import onnx
import numpy as np
import time
from enum import Enum

import torch.nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Flatten
from torch.nn import Sequential
from onnx import numpy_helper

from parse_args import parse_args, DEFAULT_NETS_DIR, DEFAULT_TINYIMAGENET_DIR


# # Check if the path was added successfully
# print("sys.path:", sys.path)

class Network(list):
    def __init__(self, net_name=None, input_name=None, input_shape=None, torch_net=None):
        super().__init__()
        self.net_name = net_name
        self.input_name = input_name
        self.input_shape = input_shape
        self.torch_net = torch_net

    def similar(self):
        res = Network()
        res.net_name = self.net_name
        res.input_name = self.input_name
        res.input_shape = self.input_shape
        res.torch_net = self.torch_net
        return res


class Layer:
    def __init__(self, weight=None, bias=None, type=None, identifier=0, parents=None):
        self.weight = weight
        self.bias = bias
        self.type = type
        self.identifier = identifier
        self.parents = parents
        self.children = []
        self.new_identifier = None

class LayerType(Enum):
    Conv2D = 1
    Linear = 2
    ReLU = 3
    Flatten = 4
    MaxPool1D = 5
    Normalization = 6
    NoOp = 7
    Concat = 8
    Sigmoid = 9
    Add = 10
    Input = 11
    Reshape = 12

class Add(torch.nn.Module):
    def forward(self, x1, x2):
        return x1 + x2

class Reshape(Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape[1:])

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)


class TransformedNet(Module):
    def __init__(self, layers, ignore_last_layer=False):
        super().__init__()
        constructed_layers = []
        self.concat_layers = dict()
        self.add_layers = dict()
        self.conv_layer_count = 0
        self.linear_layer_count = 0
        # Initially the there will be only one channel.
        self.last_conv_layer_channels = 1
        self.layers = layers
        for layer in layers:
            if layer.type == LayerType.Linear:
                shape = layer.weight.shape
                if shape is None:
                    raise ValueError("Shape of the linear layer should be not null")
                input_len = shape[1]
                output_len = shape[0]
                # Built on the meta device: the ONNX weights replace both parameters below,
                # so a random init would only be wasted work inside the timed run.
                constructed_layers.append(Linear(input_len, output_len, device='meta'))
                constructed_layers[-1].weight = torch.nn.Parameter(layer.weight)
                constructed_layers[-1].bias = torch.nn.Parameter(layer.bias)
            elif layer.type == LayerType.Conv2D:
                # self.conv_layer_count += 1
                kernel_size = layer.kernel_size
                padding = layer.padding
                stride = layer.stride
                dilation = layer.dilation
                in_channels = self.last_conv_layer_channels
                out_channels = layer.weight.shape[0]
                self.last_conv_layer_channels = out_channels
                constructed_layers.append(Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, dilation=dilation, device='meta'))
                constructed_layers[-1].weight = torch.nn.Parameter(layer.weight)
                constructed_layers[-1].bias = torch.nn.Parameter(layer.bias)
            elif layer.type == LayerType.ReLU:
                constructed_layers.append(ReLU())
            elif layer.type == LayerType.Sigmoid:
                constructed_layers.append(Sigmoid())
            elif layer.type == LayerType.Flatten:
                constructed_layers.append(Flatten(start_dim=1))
            elif layer.type == LayerType.Reshape:
                constructed_layers.append(Reshape(layer.shape))
            elif layer.type == LayerType.Concat:
                constructed_layers.append(None)
                self.concat_layers[layer.identifier] = layer.parents
            elif layer.type == LayerType.Add:
                constructed_layers.append(None)
                self.add_layers[layer.identifier] = layer.parents
            else:
                raise NotImplementedError("Layer conversion of type {} is not supported".format(layer.type))
        if ignore_last_layer:
            _ = constructed_layers.pop()
        self.model = Sequential(*constructed_layers)
        # self.model = constructed_layers
        

    def forward(self, x):
        outputs = [x]
        for i, layer in enumerate(self.model):
            if layer is not None:
                output = layer(outputs[self.layers[i].parents[0]])
                outputs.append(output)
            elif i+1 in self.concat_layers:
                output = torch.cat((outputs[self.concat_layers[i+1][0]], outputs[self.concat_layers[i+1][1]]), dim=self.layers[i].axis)
                outputs.append(output)
            elif i+1 in self.add_layers:
                output = outputs[self.add_layers[i+1][0]] + outputs[self.add_layers[i+1][1]]
                outputs.append(output)
            else:
                raise ValueError("Layer not found in add or concat layers")
                # output = layer(outputs[self.layers[i].parents[0]])
                # outputs.append(output)
            # else:
            #     output = torch.cat((outputs[self.concat_layers[i+1][0]], outputs[self.concat_layers[i+1][1]]), dim=1)
            #     outputs.append(output)
        return outputs[-1]


def convert_model(parsed_net, remove_last_layer=False):
    return TransformedNet(parsed_net, remove_last_layer)

class _LazyArrays(dict):
    """Initializer arrays for shape folding, converted on first lookup; weights are never read."""

    def __init__(self, initializers):
        super().__init__()
        self._protos = {init.name: init for init in initializers}

    def __contains__(self, name):
        return dict.__contains__(self, name) or name in self._protos

    def __missing__(self, name):
        value = numpy_helper.to_array(self._protos[name])
        self[name] = value
        return value

    def get(self, name, default=None):
        return self[name] if name in self else default


def _shape_value(node, values, shapes):
    op = node.op_type
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    if op == 'Constant':
        if 'value' in attrs:
            return numpy_helper.to_array(attrs['value'])
        for key in ('value_int', 'value_ints', 'value_float', 'value_floats'):
            if key in attrs:
                return np.asarray(attrs[key])
        raise ValueError(f"Unsupported Constant {node.name!r}")
    if op == 'Shape':
        shape = shapes.get(node.input[0])
        if shape is None:
            raise ValueError(f"Shape {node.name!r} has an unresolved input")
        return np.asarray(shape[attrs.get('start', 0):attrs.get('end')], dtype=np.int64)
    if op not in ('Gather', 'Unsqueeze', 'Squeeze', 'Concat', 'Mul', 'Add', 'Sub', 'Div', 'Cast'):
        return None
    if not all(name in values for name in node.input):
        if op in ('Gather', 'Unsqueeze', 'Squeeze', 'Cast'):
            raise ValueError(f"Unsupported activation operation {op} {node.name!r}")
        return None
    args = [values[name] for name in node.input]
    if op == 'Gather':
        return np.take(args[0], args[1], axis=attrs.get('axis', 0))
    if op in ('Unsqueeze', 'Squeeze'):
        axes = args[1].tolist() if len(args) > 1 else attrs.get('axes')
        axes = tuple(axes) if axes is not None else None
        return np.expand_dims(args[0], axes) if op == 'Unsqueeze' else np.squeeze(args[0], axes)
    if op == 'Concat':
        return np.concatenate(args, axis=attrs['axis'])
    if op == 'Cast':
        return args[0].astype(onnx.helper.tensor_dtype_to_np_dtype(attrs['to']))
    if op == 'Div':
        result = np.divide(*args)
        return np.trunc(result).astype(args[0].dtype) if np.issubdtype(args[0].dtype, np.integer) else result
    return {'Mul': np.multiply, 'Add': np.add, 'Sub': np.subtract}[op](*args)


def _reshape_shape(node, source, values):
    target = values.get(node.input[1])
    if target is None or target.ndim != 1 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError(f"Reshape {node.name!r} requires a resolved integer shape")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    shape = target.tolist()
    if not attrs.get('allowzero', 0):
        for i, dim in enumerate(shape):
            if dim == 0:
                if i >= len(source):
                    raise ValueError(f"Reshape {node.name!r} has an invalid zero dimension")
                shape[i] = source[i]
    size = math.prod(source)
    if shape.count(-1) > 1 or any(dim < -1 or dim == 0 for dim in shape):
        raise ValueError(f"Reshape {node.name!r} has invalid dimensions {shape}")
    if -1 in shape:
        known = -math.prod(shape)
        if size % known:
            raise ValueError(f"Reshape {node.name!r} changes the neuron count")
        shape[shape.index(-1)] = size // known
    if math.prod(shape) != size or len(shape) not in (2, 4) or shape[0] != 1:
        raise ValueError(f"Unsupported Reshape {node.name!r}: {source} -> {shape}")
    return shape


#FEAT: weight cast for --dtype float64.
def _default_float(t):
    """Float weights in the run's dtype (fp32 weights, or exact fp64 copies under --dtype float64)."""
    return t.to(torch.get_default_dtype()) if t.is_floating_point() else t


def parse_onnx_layers(net):
    # Shapes are tracked node by node below, as constraintflow-gpu's parser does, instead of
    # running onnx.shape_inference over the whole graph on every load.
    input_dims = net.graph.input[0].type.tensor_type.shape.dim
    shapes = {net.graph.input[0].name: [d.dim_value if d.dim_value > 0 else 1 for d in input_dims]}
    input_shape = tuple(shapes[net.graph.input[0].name])
    # Each weight is converted once (model_name_to_val_dict); shape folding reads only the constants it needs.
    values = _LazyArrays(net.graph.initializer)
    # Create the new Network object
    layers = Network(input_name=net.graph.input[0].name, input_shape=input_shape)
    num_layers = len(net.graph.node) 
    #FEAT: float weights follow the run dtype.
    model_name_to_val_dict = {init_vals.name: _default_float(torch.tensor(numpy_helper.to_array(init_vals))) for init_vals in
                              net.graph.initializer}

    names_hash = dict()
    index = 0
    names_hash[net.graph.input[0].name] = index
    parents = dict()

    layer = Layer(type=LayerType.Input, parents = [])
    layers.append(layer)
    for cur_layer in range(num_layers):
        node = net.graph.node[cur_layer]
        operation = node.op_type
        nd_inps = node.input
        value = _shape_value(node, values, shapes)
        if value is not None:
            values[node.output[0]] = value
            shapes[node.output[0]] = list(value.shape)
            continue
        index += 1
        if operation == 'MatMul':
            if nd_inps[0] not in names_hash or nd_inps[1] not in model_name_to_val_dict:
                raise ValueError(f"MatMul {node.name!r} requires activation @ constant weight")
            weight = model_name_to_val_dict[nd_inps[1]]
            if weight.ndim != 2 or len(shapes[nd_inps[0]]) != 2:
                raise ValueError(f"MatMul {node.name!r} requires two matrix operands")
            names_hash[node.output[0]] = index
            parents[index] = [names_hash[nd_inps[0]]]
            shapes[node.output[0]] = [shapes[nd_inps[0]][0], weight.shape[1]]
            layer = Layer(weight=weight.T.contiguous(), bias=weight.new_zeros(weight.shape[1]),
                          type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)

        elif operation == 'Reshape':
            result_shape = _reshape_shape(node, shapes[nd_inps[0]], values)
            shapes[node.output[0]] = result_shape
            names_hash[node.output[0]] = index
            parents[index] = [names_hash[nd_inps[0]]]
            layer = Layer(type=LayerType.Reshape, identifier=index, parents=parents[index])
            layer.shape = result_shape
            layers.append(layer)

        elif operation == 'Conv':
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            if isinstance(nd_inps[0], str):
                w_key = None
                b_key = None 
                for i in range(len(nd_inps)):
                    if 'weight' in nd_inps[i]:
                        w_key = nd_inps[i]
                    if 'bias' in nd_inps[i]:
                        b_key = nd_inps[i]
                if b_key == None:
                    layer = Layer(weight=model_name_to_val_dict[w_key], bias=torch.zeros(model_name_to_val_dict[w_key].shape[0]), type=LayerType.Conv2D, identifier=index, parents=parents[index])
                else:
                    layer = Layer(weight=model_name_to_val_dict[w_key], bias=model_name_to_val_dict[b_key], type=LayerType.Conv2D, identifier=index, parents=parents[index])

            else:
                layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Conv2D, identifier=index, parents=parents[index])
            layer.kernel_size = (node.attribute[2].ints[0], node.attribute[2].ints[1])
            layer.padding = (node.attribute[3].ints[0], node.attribute[3].ints[1])
            layer.stride = (node.attribute[4].ints[0], node.attribute[4].ints[1])
            layer.dilation = (1, 1)
            n, _c, h, w = shapes[nd_inps[0]]
            out_channels, _in_channels, k_h, k_w = layer.weight.shape
            shapes[node.output[0]] = [n, out_channels,
                                      (h + 2 * layer.padding[0] - k_h) // layer.stride[0] + 1,
                                      (w + 2 * layer.padding[1] - k_w) // layer.stride[1] + 1]
            layers.append(layer)

        elif operation == 'Gemm':
            if names_hash[str(net.graph.node[cur_layer].input[0])] in parents and layers[parents[names_hash[str(net.graph.node[cur_layer].input[0])]][0]-1].type==LayerType.Conv2D:
                parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
                layer = Layer(type=LayerType.Flatten, identifier=index, parents=parents[index])
                layers.append(layer)
                index += 1
                parents[index] = [index-1]

            else:
                parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            
            names_hash[str(net.graph.node[cur_layer].output[0])] = index

            # Making some weird assumption that the weight is always 1th index
            layer = Layer(weight=model_name_to_val_dict[nd_inps[1]], bias=(model_name_to_val_dict[nd_inps[2]]),
                          type=LayerType.Linear, identifier=index, parents=parents[index])
            layers.append(layer)
            shapes[node.output[0]] = [1, layer.weight.shape[0]]

        elif operation == 'Relu':
            shapes[node.output[0]] = shapes[nd_inps[0]]
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]

            final_relu = True
            layers.append(Layer(type=LayerType.ReLU, identifier=index, parents=parents[index]))

        elif operation == 'Sigmoid':
            shapes[node.output[0]] = shapes[nd_inps[0]]
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]

            layers.append(Layer(type=LayerType.Sigmoid, identifier=index, parents=parents[index]))

        elif operation == 'Concat':
            if len(nd_inps) != 2 or not all(name in names_hash for name in nd_inps):
                raise ValueError(f"Concat {node.name!r} requires two activation inputs")
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            axis = attrs['axis']
            rank = len(shapes[nd_inps[0]])
            if axis < 0:
                axis += rank
            if not 0 < axis < rank:
                raise ValueError(f"Unsupported Concat axis {axis}")
            names_hash[node.output[0]] = index
            parents[index] = [names_hash[name] for name in nd_inps]
            layer = Layer(type=LayerType.Concat, identifier=index, parents=parents[index])
            layer.axis = axis
            layers.append(layer)
            result_shape = list(shapes[nd_inps[0]])
            result_shape[axis] += shapes[nd_inps[1]][axis]
            shapes[node.output[0]] = result_shape

        elif operation == 'Flatten':
            axis = next((a.i for a in node.attribute if a.name == 'axis'), 1)
            if axis not in (1, 1 - len(shapes[nd_inps[0]])):
                raise ValueError(f"Unsupported Flatten axis {axis}")
            names_hash[str(net.graph.node[cur_layer].output[0])] = index
            parents[index] = [names_hash[str(net.graph.node[cur_layer].input[0])]]
            layer = Layer(type=LayerType.Flatten, identifier=index, parents=parents[index])
            shapes[node.output[0]] = [shapes[nd_inps[0]][0], math.prod(shapes[nd_inps[0]][1:])]
            layers.append(layer)

        elif operation == 'Add':
            shapes[node.output[0]] = shapes[nd_inps[0]]
            if nd_inps[1] not in model_name_to_val_dict:
                names_hash[str(net.graph.node[cur_layer].output[0])] = index
                parent1 = names_hash[nd_inps[0]]
                parent2 = names_hash[nd_inps[1]]
                parents[index] = [parent1, parent2]
                layer = Layer(weight=None, type=LayerType.Add, identifier=index, parents=parents[index])
                layers.append(layer)
            else:
                index -= 1
                names_hash[str(net.graph.node[cur_layer].output[0])] = index
                layer = layers[-1]
                consumers = sum(name == nd_inps[0] for n in net.graph.node for name in n.input)
                if layer.identifier != names_hash[nd_inps[0]] or layer.type != LayerType.Linear or consumers != 1:
                    raise ValueError(f"Add {node.name!r} cannot be folded into a linear bias")
                layer.bias = layer.bias + model_name_to_val_dict[nd_inps[1]]

        else:
            index -= 1
            if len(net.graph.node[cur_layer].input)>0:
                if str(net.graph.node[cur_layer].input[0]) in names_hash:
                    names_hash[str(net.graph.node[cur_layer].output[0])] = names_hash[str(net.graph.node[cur_layer].input[0])]
                    shapes[node.output[0]] = shapes[nd_inps[0]]

    # The final most layer is relu and no linear layer after that
    # remove the linear layer (Hack find better solutions).
    # if final_relu is True:
    #     layers.pop()
    layers = post_process(layers)
    # for i, layer in enumerate(layers):
        # print(i, layer.type, layer.identifier, layer.parents)÷
    layers = layers[1:]
    return layers


def post_process(layers):
    identifier_to_index = dict()
    for i, layer in enumerate(layers):
        identifier_to_index[layer.identifier] = i
    for i, layer in enumerate(layers):
        for j, parent in enumerate(layer.parents):
            layers[identifier_to_index[parent]].children.append(layer.identifier)
    
    queue = deque([layers[identifier_to_index[0]]])
    visited = set()
    layer_num = 0
    new_layers = layers.similar()
    while queue:
        layer = queue.popleft()
        if layer in visited:
            continue
        layer.new_identifier = layer_num
        new_layers.append(layer)
        layer_num += 1
        visited.add(layer)
        for child in layer.children:
            if child not in visited:
                parent_visited = True 
                for parent in layers[identifier_to_index[child]].parents:
                    if layers[identifier_to_index[parent]] not in visited:
                        parent_visited = False
                        break
                if parent_visited:
                    queue.append(layers[identifier_to_index[child]])
    
    for i, layer in enumerate(new_layers):
        parents = []
        for parent in layer.parents:
            parents.append(layers[parent].new_identifier)
        layer.parents = parents
        children = []
        for child in layer.children:
            children.append(layers[child].new_identifier)
        layer.children = children
        layer.identifier = layer.new_identifier
    return new_layers

def get_pytorch_from_onnx(onnx_file):
    net_onnx = onnx.load(onnx_file)
    return Sequential(convert_model(parse_onnx_layers(net_onnx)), Flatten(start_dim=1))

def perturb_model_(model, std, rng):
    """In-place iid Gaussian noise (given std) added to every parameter, to test whether
    weight values (not just shapes) affect measured runtime. Shapes/dtypes are unchanged."""
    if std <= 0:
        return model
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.from_numpy(rng.standard_normal(p.shape)).to(dtype=p.dtype, device=p.device)
            p.add_(noise * std)
    return model

def get_spec_matrix(X, y, num_classes):
    with torch.no_grad():
        c = (torch.eye(num_classes).type_as(X)[y].unsqueeze(1)
            - torch.eye(num_classes).type_as(X).unsqueeze(0))
        I = (~(y.unsqueeze(1) == torch.arange(num_classes).type_as(y).unsqueeze(0)))
        c = (c[I].view(X.size(0), num_classes - 1, num_classes))
    return c

def get_precision(lb):
    verified = (lb >= 0).all(dim=1)
    precision = verified.sum() / verified.shape[0]
    return precision

def get_dataset(batch_size: int, dataset: str, train: bool = False):
    if dataset == "mnist":
        transform = transforms.ToTensor()  # keep 28x28
        data = datasets.MNIST(root=".", train=train, download=True, transform=transform)
    elif dataset == "cifar10" or dataset == "cifar":
        transform = transforms.ToTensor()  # keep 32x32
        data = datasets.CIFAR10(root=".", train=train, download=True, transform=transform)
    elif dataset == "tinyimagenet":
        train = True
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # TinyImageNet images are 64x64
            transforms.ToTensor(),
        ])
        root_dir = DEFAULT_TINYIMAGENET_DIR
        split = "train" if train else "test"
        data_dir = os.path.join(root_dir, split)
        if train:
            data = datasets.ImageFolder(root=data_dir, transform=transform)
        else:
            # TinyImageNet test: all images in one folder
            from torchvision.datasets.folder import default_loader
            class TinyImageNetTest(torch.utils.data.Dataset):
                def __init__(self, root, transform=None):
                    self.root = root
                    self.transform = transform
                    self.loader = default_loader
                    self.images = sorted(os.listdir(root))
                def __len__(self):
                    return len(self.images)
                def __getitem__(self, idx):
                    path = os.path.join(self.root, self.images[idx])
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    return img, -1
            data = TinyImageNetTest(data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    image, label = next(iter(dataloader))

    # ensure labels are a tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    return image, label

def run_exp(onnx_file, N=1, eps = 0.01, dataset="mnist", program='backward', nets_dir=DEFAULT_NETS_DIR, device='cpu', warmup=0, repeat=1, perturb_eps=0.0, perturb_seed=0, num_images=0, results_file=None, profile_comm=False):
    import numpy as np
    perturb_rng = np.random.default_rng(perturb_seed) if perturb_eps > 0 else None
    mnist = dataset == "mnist"
    mnist_mean = torch.tensor([0.1307]).reshape(1, 1, 1, 1)
    mnist_std = torch.tensor([0.3081]).reshape(1, 1, 1, 1)
    cifar10_mean = torch.tensor([0.4914, 0.4822, 0.4465]).reshape(1, -1, 1, 1)
    cifar10_std = torch.tensor([0.2023, 0.1994, 0.2010]).reshape(1, -1, 1, 1)

    print(f'Experiment:')
    print(f'Network: {onnx_file}')
    print(f'Batch size: {N}')
    print(f'Epsilon: {eps}')
    print(f'TF32: {"disabled (CF_DISABLE_TF32)" if DISABLE_TF32 else "PyTorch default"}')
    #FEAT: report the run dtype.
    print(f'dtype: {torch.get_default_dtype()}')
    print(f'Dataset: {"mnist" if mnist else "cifar"}')
    print()
    folder = nets_dir
    total = num_images + warmup * N if num_images else N * (warmup + 1)
    if mnist:
        onnx_file = os.path.join(folder, 'mnist', onnx_file)
        test_data = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
        true_label = test_data.targets[:total]
        image = test_data.data[:total].view(-1,1,28,28)
    elif dataset == "cifar" or dataset == "cifar10":
        onnx_file = os.path.join(folder, 'cifar', onnx_file)
        test_data = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
        true_label = test_data.targets[:total]
        true_label = torch.tensor(true_label)
        image = torch.from_numpy(test_data.data[:total]).permute(0, 3, 1, 2).view(-1, 3, 32, 32)
    elif dataset == "tinyimagenet":
        onnx_file = os.path.join(folder, onnx_file)
        image, true_label = get_dataset(total, dataset, train=True)
        # image = image[:, :, :56, :56]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if dataset != 'tinyimagenet':
        #FEAT: images follow the run dtype instead of hardcoded fp32.
        image = image.to(torch.get_default_dtype()) / 255.0
    else:
        image = image[:, :, :56, :56]
        #FEAT: images follow the run dtype instead of hardcoded fp32.
        image = image.to(torch.get_default_dtype()) / 255.0
    # Images stay on the host; each run moves only its own batch, as the other harnesses do.
    IMG_ALL = image
    LABEL_ALL = true_label

    if num_images:
        return _run_image_batches(IMG_ALL, LABEL_ALL, onnx_file, N, eps, dataset, mnist, program,
                                  device, warmup, num_images, results_file,
                                  mnist_mean, mnist_std, cifar10_mean, cifar10_std, profile_comm)

    for i in range(warmup):
        warmup_start = time.time()
        IMG = IMG_ALL[(i + 1) * N : (i + 2) * N].to(device)
        true_label = LABEL_ALL[(i + 1) * N : (i + 2) * N].to(device)

        original_image = copy.deepcopy(IMG)

        if mnist:
            mean = mnist_mean
            std = mnist_std
        else:
            mean = cifar10_mean
            std = cifar10_std

        mean = mean.to(device).expand(IMG.shape)
        std = std.to(device).expand(IMG.shape)
        norm = float("inf")
        x_L = torch.clip(IMG - eps, min=0., max=1.)

        x_L = (x_L - mean) / std

        x_U = torch.clip(IMG + eps, min=0., max=1.)
        x_U = (x_U - mean) / std


        ptb = PerturbationLpNorm(norm = norm, x_L = x_L, x_U = x_U)
        image = BoundedTensor(IMG, ptb)

        if dataset == 'tinyimagenet':
            num_classes = 200
        else:
            num_classes = 10
        C = get_spec_matrix(image, true_label, num_classes)

        with torch.no_grad():
            model = get_pytorch_from_onnx(onnx_file)
            perturb_model_(model, perturb_eps, perturb_rng)
            lirpa_model = BoundedModule(model, original_image, device=image.device)
            lb, ub = lirpa_model.compute_bounds(x=(image,), method=program, C=C)

        if device == 'cuda':
            torch.cuda.synchronize()
        print(f'Warmup run {i + 1}/{warmup}: {time.time() - warmup_start} s')
        ### Free Memory ####
        lb, ub = lb.detach().cpu(), ub.detach().cpu()
        del lirpa_model, model, lb, ub
        del IMG, true_label, original_image, mean, std, x_L, x_U, ptb, image, C
        gc.collect()
        # Not needed for the per-run peak (max_memory_allocated counts live tensors only);
        # it only made the next run re-cudaMalloc its whole working set.
        # if device == 'cuda':
        #     torch.cuda.empty_cache()

    # Freeze the startup heap so the between-run gc.collect() only scans per-run garbage.
    gc.collect()
    gc.freeze()

    ####################################################################


    lb = ub = None
    run_times = []
    run_peaks = []
    for i in range(repeat):
        resident = 0
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
        start_time = time.time()
        # This run's input box and spec are built inside the timer, as in the other harnesses.
        IMG = IMG_ALL[:N].to(device)
        true_label = LABEL_ALL[:N].to(device)

        original_image = copy.deepcopy(IMG)

        if mnist:
            mean = mnist_mean
            std = mnist_std
        else:
            mean = cifar10_mean
            std = cifar10_std

        mean = mean.to(device).expand(IMG.shape)
        std = std.to(device).expand(IMG.shape)
        norm = float("inf")
        x_L = torch.clip(IMG - eps, min=0., max=1.)

        x_L = (x_L - mean) / std

        x_U = torch.clip(IMG + eps, min=0., max=1.)
        x_U = (x_U - mean) / std


        ptb = PerturbationLpNorm(norm = norm, x_L = x_L, x_U = x_U)
        image = BoundedTensor(IMG, ptb)

        if dataset == 'tinyimagenet':
            num_classes = 200
        else:
            num_classes = 10
        C = get_spec_matrix(image, true_label, num_classes)

        with torch.no_grad():
            model = get_pytorch_from_onnx(onnx_file)
            perturb_model_(model, perturb_eps, perturb_rng)
            lirpa_model = BoundedModule(model, original_image, device=image.device)
            lb, ub = lirpa_model.compute_bounds(x=(image,), method=program, C=C)

        if device == 'cuda':
            torch.cuda.synchronize()
        run_time = time.time() - start_time
        run_peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None
        run_times.append(run_time)
        run_peaks.append(run_peak)
        if repeat > 1:
            if device == 'cuda':
                print(f'Run {i + 1}/{repeat}: {run_time} s, Peak GPU memory: {run_peak} bytes, Resident GPU memory: {resident} bytes')
            else:
                print(f'Run {i + 1}/{repeat}: {run_time} s')

        ### Free Memory ####
        lb, ub = lb.detach().cpu(), ub.detach().cpu()
        del lirpa_model, model
        del IMG, true_label, original_image, mean, std, x_L, x_U, ptb, image, C
        gc.collect()
        # Not needed for the per-run peak (max_memory_allocated counts live tensors only);
        # it only made the next run re-cudaMalloc its whole working set.
        # if device == 'cuda':
        #     torch.cuda.empty_cache()

    t = sum(run_times) / len(run_times)
    print(f'Average verification time: {t}')
    if device == 'cuda':
        print(f'Peak GPU memory: {max(run_peaks)} bytes')
    # print(f'Precision: {precision:.4f}')

    # Full float32 precision (9 significant digits) and no '...' elision, so the harness
    # can parse the exact bounds into its CSV.
    torch.set_printoptions(precision=8, sci_mode=True, threshold=sys.maxsize)
    print(f'Lower bound: {lb}')
    print(f'Upper bound: {ub}')
    torch.set_printoptions(profile="default")
    print()
    print()
    return lb, ub, t

def _certify_batch(IMG_ALL, LABEL_ALL, lo, N, onnx_file, eps, dataset, mnist, program, device,
                   mnist_mean, mnist_std, cifar10_mean, cifar10_std):
    """One run of the timed region, as in run_exp's --repeat loop, on images [lo, lo + N)."""
    IMG = IMG_ALL[lo : lo + N].to(device)
    true_label = LABEL_ALL[lo : lo + N].to(device)

    original_image = copy.deepcopy(IMG)

    if mnist:
        mean = mnist_mean
        std = mnist_std
    else:
        mean = cifar10_mean
        std = cifar10_std

    mean = mean.to(device).expand(IMG.shape)
    std = std.to(device).expand(IMG.shape)
    norm = float("inf")
    x_L = torch.clip(IMG - eps, min=0., max=1.)

    x_L = (x_L - mean) / std

    x_U = torch.clip(IMG + eps, min=0., max=1.)
    x_U = (x_U - mean) / std


    ptb = PerturbationLpNorm(norm = norm, x_L = x_L, x_U = x_U)
    image = BoundedTensor(IMG, ptb)

    if dataset == 'tinyimagenet':
        num_classes = 200
    else:
        num_classes = 10
    C = get_spec_matrix(image, true_label, num_classes)

    with torch.no_grad():
        model = get_pytorch_from_onnx(onnx_file)
        lirpa_model = BoundedModule(model, original_image, device=image.device)
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=program, C=C)

    if device == 'cuda':
        torch.cuda.synchronize()
    return lb, ub


def _run_image_batches(IMG_ALL, LABEL_ALL, onnx_file, N, eps, dataset, mnist, program, device,
                       warmup, num_images, results_file,
                       mnist_mean, mnist_std, cifar10_mean, cifar10_std, profile_comm=False):
    """The --num_images protocol: warmups on the images after the evaluated ones, then one
    timed run per batch, each writing one JSON line after its timer stops. No error
    handling: a failing batch kills the process, and finished batches stay in the file."""
    import json
    #FEAT: --profile_comm profiles each timed region.
    from comm_profile import CommProfile

    args = (onnx_file, eps, dataset, mnist, program, device,
            mnist_mean, mnist_std, cifar10_mean, cifar10_std)

    def write(f, record):
        f.write(json.dumps(record) + "\n")
        f.flush()

    with open(results_file, "w") as out:
        for j in range(warmup):
            lo = num_images + j * N
            warmup_start = time.time()
            lb, ub = _certify_batch(IMG_ALL, LABEL_ALL, lo, N, *args)
            warmup_time = time.time() - warmup_start
            print(f'Warmup run {j + 1}/{warmup}: {warmup_time} s')
            write(out, {"kind": "warmup", "index": j, "images": [lo, lo + N], "time_s": warmup_time})
            del lb, ub
            gc.collect()

        # Freeze the startup heap so the between-run gc.collect() only scans per-run garbage.
        gc.collect()
        gc.freeze()

        n_batches = num_images // N
        for i in range(n_batches):
            lo = i * N
            print(f'=== batch {i} start ===')
            resident = None
            #FEAT: profiler on around the timer; comm.region() is exactly the timed region.
            comm = CommProfile(profile_comm)
            with comm:
                if device == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    resident = torch.cuda.memory_allocated()
                start_time = time.time()
                # This run's input box and spec are built inside the timer, as in the other harnesses.
                with comm.region():
                    lb, ub = _certify_batch(IMG_ALL, LABEL_ALL, lo, N, *args)
                run_time = time.time() - start_time
            # Per-run peaks exist only on the GPU; a CPU process's peak RSS is lifetime-wide.
            peak = torch.cuda.max_memory_allocated() if device == 'cuda' else None

            write(out, {"kind": "batch", "batch": i, "images": [lo, lo + N], comm.time_key: run_time,
                        "peak_bytes": peak, "resident_bytes": resident,
                        "lower": lb.detach().cpu().tolist(), "upper": ub.detach().cpu().tolist(),
                        **comm.metrics()})
            comm.save_trace(results_file, i)
            print(f'Run {i + 1}/{n_batches}: {run_time} s, Peak memory: {peak} bytes, Resident memory: {resident} bytes')
            print(f'=== batch {i} end ===', flush=True)
            del lb, ub
            gc.collect()


def experiments(networks, N, eps, mnist):
    for i, network in enumerate(networks):
        run_exp(network, N, eps, mnist)



def get_all_networks(folder, prefixes, network_format):
    networks = [
        f for f in os.listdir(folder)
        if f.startswith(prefixes) and os.path.isfile(os.path.join(folder, f))
    ]
    networks.sort()
    return networks



if __name__ == '__main__':
    args = parse_args()

    if args.logs != 'stdout':
        sys.stdout = open(args.logs, 'w')

    if args.program == 'deeppoly':
        program = 'backward'
    elif args.program == 'crownibp':
        program = 'crown-ibp'
    elif args.program == 'ibp':
        program = 'ibp'
    else:
        raise ValueError(f"Unknown program: {args.program}")

    if args.device == 'gpu' and not torch.cuda.is_available():
        raise RuntimeError("--device gpu requested but CUDA is not available.")
    torch_device = 'cuda' if args.device == 'gpu' else 'cpu'
    #FEAT: --dtype float64 sets the default dtype before anything is built.
    if args.dtype == 'float64':
        # Precision oracle: weights, images, box and every bound computed in fp64.
        torch.set_default_dtype(torch.float64)

    if args.single_experiment:
        if args.num_images and (args.num_images % args.batch_size or not args.results_file):
            raise ValueError("--num_images must be a positive multiple of --batch_size and needs --results_file")
        if args.profile_comm and not args.num_images:
            raise ValueError("--profile_comm requires --num_images")
        run_exp(f'{args.network}.{args.network_format}', args.batch_size, args.eps, dataset = args.dataset, program=program, nets_dir=args.nets_dir, device=torch_device, warmup=args.warmup, repeat=args.repeat, num_images=args.num_images, results_file=args.results_file, profile_comm=args.profile_comm)
    else:
        if args.exp_name == 'eran':
            if args.dataset == 'mnist':
                folder = os.path.join(args.nets_dir, 'eran', 'mnist')
                if args.network_type == 'fcn':
                    prefixes = ('mnist_relu', 'ffnn', 'mnist-net')
                elif args.network_type == 'cnn':
                    prefixes = ('conv', 'skip', 'mnist_0')
            elif args.dataset == 'cifar':
                folder = os.path.join(args.nets_dir, 'eran', 'cifar')
                if args.network_type == 'fcn':
                    prefixes = ('cifar_relu', 'ffnn')
                elif args.network_type == 'cnn':
                    prefixes = ('conv', 'cifar_conv')
            elif args.dataset == 'imagenet':
                pass
        networks = get_all_networks(folder, prefixes, args.network_format)
        experiments(networks, args.batch_size, args.eps, args.dataset=='mnist', args.network_type=='fcn')


