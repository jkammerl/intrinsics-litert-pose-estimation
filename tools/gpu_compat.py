"""Rewrites ops in the FoundationPose TFLite models that GPUs compute wrongly.

LiteRT's Metal GPU accelerator (macOS) computes two op patterns of the
scorer wrongly, so that the GPU check at compilation (validate_gpu) rejected
it:
  * SPLIT, whose outputs are concatenated on another axis: the concatenation
    reads the wrong data. Each SPLIT becomes one SLICE per output.
  * BATCH_MATMUL with adj_x and a left operand of shape [..., K, 1]: its
    transpose is only a reshape, so it becomes a RESHAPE to [..., 1, K] and a
    BATCH_MATMUL without adj_x.
Both rewrites only move data, so the results on the CPU stay the same.

convert.py applies them to the models it writes. For models converted
before, rewrite them in place (models without these ops are left as they
are):

  python gpu_compat.py --models=../models
"""

import argparse
import glob
import os

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

_OPS = schema.BuiltinOperator


def _builtin(model, op):
  code = model.operatorCodes[op.opcodeIndex]
  return max(code.builtinCode, code.deprecatedBuiltinCode)


def _opcode_index(model, builtin):
  for i, code in enumerate(model.operatorCodes):
    if max(code.builtinCode, code.deprecatedBuiltinCode) == builtin:
      return i
  code = schema.OperatorCodeT()
  code.builtinCode = builtin
  code.deprecatedBuiltinCode = min(builtin, 127)
  code.version = 1
  model.operatorCodes.append(code)
  return len(model.operatorCodes) - 1


def _add_tensor(subgraph, name, shape, dtype, buffer=0):
  tensor = schema.TensorT()
  tensor.name = name
  tensor.shape = list(shape)
  tensor.type = dtype
  tensor.buffer = buffer
  subgraph.tensors.append(tensor)
  return len(subgraph.tensors) - 1


def _add_int32_constant(model, subgraph, name, values):
  buffer = schema.BufferT()
  buffer.data = np.asarray(values, np.int32).tobytes()
  model.buffers.append(buffer)
  return _add_tensor(subgraph, name, [len(values)], schema.TensorType.INT32,
                     len(model.buffers) - 1)


def _split_to_slices(model, subgraph):
  slice_code = None
  ops, count = [], 0
  for op in subgraph.operators:
    if _builtin(model, op) != _OPS.SPLIT:
      ops.append(op)
      continue
    if slice_code is None:
      slice_code = _opcode_index(model, _OPS.SLICE)
    axis_tensor = subgraph.tensors[op.inputs[0]]
    axis = int(np.frombuffer(
        bytes(model.buffers[axis_tensor.buffer].data), np.int32)[0])
    rank = len(subgraph.tensors[op.inputs[1]].shape)
    axis %= rank
    begin = 0
    for k, output in enumerate(op.outputs):
      size = list(subgraph.tensors[output].shape)
      start = [0] * rank
      start[axis] = begin
      begin += size[axis]
      name = f"gpu_compat/split{count}_{k}"
      slice_op = schema.OperatorT()
      slice_op.opcodeIndex = slice_code
      slice_op.inputs = [
          op.inputs[1],
          _add_int32_constant(model, subgraph, name + "_begin", start),
          _add_int32_constant(model, subgraph, name + "_size", size),
      ]
      slice_op.outputs = [output]
      slice_op.builtinOptionsType = schema.BuiltinOptions.SliceOptions
      slice_op.builtinOptions = schema.SliceOptionsT()
      ops.append(slice_op)
    count += 1
  subgraph.operators = ops
  return count


def _unadjoint_batch_matmuls(model, subgraph):
  reshape_code = None
  ops, count = [], 0
  for op in subgraph.operators:
    lhs = subgraph.tensors[op.inputs[0]]
    if (_builtin(model, op) == _OPS.BATCH_MATMUL and op.builtinOptions.adjX
        and len(lhs.shape) >= 2 and lhs.shape[-1] == 1):
      if reshape_code is None:
        reshape_code = _opcode_index(model, _OPS.RESHAPE)
      shape = list(lhs.shape)
      shape[-2], shape[-1] = shape[-1], shape[-2]
      name = f"gpu_compat/batch_matmul{count}_lhs"
      reshaped = _add_tensor(subgraph, name, shape, lhs.type)
      reshape = schema.OperatorT()
      reshape.opcodeIndex = reshape_code
      reshape.inputs = [
          op.inputs[0],
          _add_int32_constant(model, subgraph, name + "_shape", shape),
      ]
      reshape.outputs = [reshaped]
      reshape.builtinOptionsType = schema.BuiltinOptions.ReshapeOptions
      reshape.builtinOptions = schema.ReshapeOptionsT()
      reshape.builtinOptions.newShape = shape
      ops.append(reshape)
      op.inputs = [reshaped] + list(op.inputs[1:])
      op.builtinOptions.adjX = False
      count += 1
    ops.append(op)
  subgraph.operators = ops
  return count


def rewrite(path):
  """Rewrites the model at `path` in place; returns the number of rewrites."""
  with open(path, "rb") as f:
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(f.read(), 0))
  count = 0
  for subgraph in model.subgraphs:
    count += _split_to_slices(model, subgraph)
    count += _unadjoint_batch_matmuls(model, subgraph)
  if count:
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    with open(path, "wb") as f:
      f.write(builder.Output())
  return count


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--models", required=True)
  args = parser.parse_args()
  for path in sorted(glob.glob(
      os.path.join(args.models, "foundationpose_*.tflite"))):
    print(f"{os.path.basename(path)}: {rewrite(path)} ops rewritten")


if __name__ == "__main__":
  main()
