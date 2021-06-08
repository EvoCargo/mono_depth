from torch.optim import SGD, Adam
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop


__optimizers__ = {
    'Adam': Adam,
    'SGD': SGD,
    'Adagrad': Adagrad,
    'Adadelta': Adadelta,
    'RMSprop': RMSprop,
}


def _get_optimizer(config, model_params):
    """
    :param config: OrderDict, {'name':?, 'params':?}
    :return: optimizer
    """
    if config['name'] not in __optimizers__:
        print('[Error] {} does not defined.'.format(config['name']))
        raise NotImplementedError

    # print(config["params"])
    return __optimizers__[config["name"]](model_params, **config["params"])
