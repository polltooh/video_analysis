import importlib
from TensorflowToolbox.utility import file_io

def f():
    global Model 
    Model = file_io.import_module_class("vgg_model", "Model")
def b():
    Model()
 
f()
b()
