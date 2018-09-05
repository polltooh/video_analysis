from vgg_model import Model
from traffic_data_ph import DataPh


def model_exc(is_train):
    infer = Model.model_infer(DataPh)
    loss = Model.model_loss(infer, DataPh)
    if is_train:
        Model.model_mini(loss)
