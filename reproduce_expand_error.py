import torch
from torch.fx import Graph, Node
from executorch.backends.vulkan.op_registry import vulkan_supported_ops, OpFeatures
from executorch.backends.vulkan import utils
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.dialects._ops import ops as exir_ops


def reproduce():
    # Mock a node for expand_copy
    # aten.expand_copy.default(self, size, implicit=False)

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        # Create fake tensor for input
        input_val = torch.zeros([1, 1, 29, 1024], dtype=torch.bool)

        # Create fake tensor for output
        output_val = torch.zeros([1, 1, 29, 1024], dtype=torch.bool)

    # Create a dummy graph and node
    graph = Graph()
    input_node = graph.placeholder("input")
    input_node.meta["val"] = input_val

    # size arg
    size_arg = [1, -1, -1, -1]

    # implicit arg
    implicit_arg = False

    # Create op node
    op_node = graph.call_function(
        exir_ops.edge.aten.expand_copy.default,
        args=(input_node, size_arg, implicit_arg),
    )
    op_node.meta["val"] = output_val

    # Get features
    op_key = exir_ops.edge.aten.expand_copy.default
    if op_key not in vulkan_supported_ops:
        print(f"Op {op_key} not registered!")
        return

    features = vulkan_supported_ops[op_key]
    print(f"Features: {features}")

    # Make op repsets
    texture_limits = utils.DEFAULT_TEXTURE_LIMITS
    op_repsets = features.make_op_repsets(op_node, texture_limits)

    print(f"Op Repsets: {op_repsets}")

    if op_repsets.any_is_empty():
        print("Op Repsets contains empty repset!")
        print(f"Args repsets empty: {op_repsets.args_repset_list.any_is_empty()}")
        print(f"Outs repsets empty: {op_repsets.outs_repset_list.any_is_empty()}")

        # Check individual repsets
        for i, repset in enumerate(op_repsets.args_repset_list.vals):
            print(f"Arg {i} repset empty: {repset.is_empty()}")
            print(f"Arg {i} repset: {repset}")

        for i, repset in enumerate(op_repsets.outs_repset_list.vals):
            print(f"Out {i} repset empty: {repset.is_empty()}")
            print(f"Out {i} repset: {repset}")


if __name__ == "__main__":
    reproduce()
