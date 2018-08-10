from TensorflowToolbox.utility import read_proto as rp
from TensorflowToolbox.utility import file_io
from net_flow import NetFlow
import sys
import yaml
import json

if __name__ == "__main__":
    if len(sys.argv) == 2:
        model_proto_file = sys.argv[1]
    else:
        model_proto_file = "model_proto.yaml"

    with open(model_proto_file, 'r') as f:
        model_params = yaml.load(f)

    print(json.dumps(model_params, indent=4))

    file_io.check_exist(model_params["train_file_name"])
    file_io.check_exist(model_params["test_file_name"])

    net = NetFlow(model_params, True, True)
    net.mainloop()
