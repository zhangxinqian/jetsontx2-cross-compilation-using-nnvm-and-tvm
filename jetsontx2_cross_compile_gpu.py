import nnvm.compiler
import nnvm.symbol as sym
import numpy as np

x = sym.Variable("x")
y = sym.Variable("y")
z = sym.elemwise_add(x, sym.sqrt(y))
compute_graph = nnvm.graph.create(z)
x_np = np.array([1, 2, 3, 4]).astype("float32")
y_np = np.array([4, 4, 4, 4]).astype("float32")
shape = (4, )
deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target="cuda", target_host= "llvm -target=aarch64-linux-gnu", shape={"x": shape}, params={"y": y_np}, dtype="float32")

x_np.tofile("./data/x.bin")
np.save("./data/x.nparray", x_np)
with open("./model/jetson_gpu.json", "w") as fo:
    fo.write(deploy_graph.json())
with open("./model/jetson_gpu.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

lib.save("./model/jetson_gpu.o")
dev_modules = lib.imported_modules
dev_modules[0].save("./model/jetson_gpu.ptx")
