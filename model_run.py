import tvm
import numpy as np

# tvm module for compiled functions.
loaded_lib = tvm.module.load("./model/jetson_gpu.so")
dev_lib = tvm.module.load("./model/jetson_gpu.ptx")
loaded_lib.import_module(dev_lib)
# json graph
loaded_json = open("./model/jetson_gpu.json").read()
# parameters in binary
loaded_params = bytearray(open("./model/jetson_gpu.params", "rb").read())
# data in numpy array
#x = np.load("./data/x.nparray.npy")
# data in binary
x = np.fromfile("./data/x.bin", dtype="float32")
x.shape = (4, )

fcreate = tvm.get_global_func("tvm.graph_runtime.create")
ctx = tvm.gpu(0)
gmodule = fcreate(loaded_json, loaded_lib, ctx.device_type, ctx.device_id)
set_input, get_output, run = gmodule["set_input"], gmodule["get_output"], gmodule["run"]
set_input("x", tvm.nd.array(x.astype('float32')))
gmodule["load_params"](loaded_params)
run()
out_shape = x.shape
out = tvm.nd.empty(out_shape, "float32")
get_output(0, out)
print(out.asnumpy())
