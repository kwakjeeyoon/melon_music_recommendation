import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_opt(config_train, model):
    """
    define optimizer & scheduler
    """
    param_groups = model.parameters()

    if config_train['name'] == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=float(config_train['lr']),
            weight_decay=float(config_train['weight_decay']),
            momentum=config_train['momentum'],
            nesterov=False,
        )

    elif config_train['name'] == "adam":
        # print(config_train['amsgrad'], type(config_train['lr']))
        optimizer = optim.Adam(
            param_groups,
            lr=float(config_train['lr']),
            weight_decay=float(config_train['weight_decay']),
            amsgrad=config_train['amsgrad']
        )
    # elif args.optimizer == "radam":
    #     optimizer = RAdam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not a valid optimizer")

    return optimizer