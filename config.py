import functools
import inspect
from yacs.config import CfgNode as CN


__all__ = ['get_cfg_defaults', 'convert_to_dict', 'override', 'configurable']

_C = CN()
_C.name = 'default-exp'
_C.description = ''
_C.out_root = 'results'
_C.n_sample = 64
_C.resolution = 256
_C.classes = []

# ------ dataset ------
_C.DATASET = CN()
_C.DATASET.name = 'DeepFashion'
_C.DATASET.roots = ['~/data/deepfashion']
_C.DATASET.sources = ['align_1.2']
_C.DATASET.xflip = False
# kwargs of data loader
_C.DATASET.pin_memory = False
_C.DATASET.num_workers = 4

# ------ Model ------
_C.MODEL = CN()
_C.MODEL.z_dim = 512
_C.MODEL.w_dim = 512

_C.MODEL.MAPPING = CN()
_C.MODEL.MAPPING.num_layers = 8
_C.MODEL.MAPPING.embed_dim = 512  # Force to zero if no label(len(classes) == 1)
_C.MODEL.MAPPING.layer_dim = 512
_C.MODEL.MAPPING.lrmul = 0.01

_C.MODEL.SYNTHESIS = CN()
_C.MODEL.SYNTHESIS.img_channels = 3
_C.MODEL.SYNTHESIS.bottom_res = 4
_C.MODEL.SYNTHESIS.pose_encoder_kwargs = CN(new_allowed=True)
_C.MODEL.SYNTHESIS.pose_encoder_kwargs.name = 'DefaultPoseEncoder'
_C.MODEL.SYNTHESIS.channel_base = 32768
_C.MODEL.SYNTHESIS.channel_max = 512

_C.MODEL.ATTENTION = CN()
_C.MODEL.ATTENTION.resolutions = []
_C.MODEL.ATTENTION.feature_types = 'relu'

_C.MODEL.DISCRIMINATOR = CN()
_C.MODEL.DISCRIMINATOR.c_dim = 0  # c_dim > 0: must equals to len(classes) and len(img_channels) == 1
_C.MODEL.DISCRIMINATOR.img_channels = [3]
_C.MODEL.DISCRIMINATOR.branch_res = 64
_C.MODEL.DISCRIMINATOR.top_res = 4
_C.MODEL.DISCRIMINATOR.channel_base = 32768
_C.MODEL.DISCRIMINATOR.channel_max = 512
_C.MODEL.DISCRIMINATOR.cmap_dim = 512
_C.MODEL.DISCRIMINATOR.mbstd_group_size = 4
_C.MODEL.DISCRIMINATOR.mbstd_num_features = 1
_C.MODEL.DISCRIMINATOR.resample_filter = [1, 3, 3, 1]

# ----- training ------
_C.TRAIN = CN()
_C.TRAIN.iteration = 80000
_C.TRAIN.batch_gpu = 16
_C.TRAIN.lrate = 0.002
_C.TRAIN.lrate_atten = 0.002
_C.TRAIN.PPL = CN()
_C.TRAIN.PPL.gain = 2
_C.TRAIN.PPL.bs_shrink = 2
_C.TRAIN.PPL.every = 4
_C.TRAIN.R1 = CN()
_C.TRAIN.R1.gamma = 10
_C.TRAIN.R1.every = 16
_C.TRAIN.style_mixing_prob = 0.9
_C.TRAIN.CKPT = CN()
_C.TRAIN.CKPT.path = ''
_C.TRAIN.CKPT.every = 2500
_C.TRAIN.CKPT.max_keep = 10
_C.TRAIN.SAMPLE = CN()
_C.TRAIN.SAMPLE.every = 1000


_C.ADA = CN()
_C.ADA.enabled = False
_C.ADA.target = 0.6
_C.ADA.p = 0.0
_C.ADA.interval = 4
_C.ADA.kimg = 500
_C.ADA.KWARGS = CN()
_C.ADA.KWARGS.xflip = 1
_C.ADA.KWARGS.rotate90 = 1
_C.ADA.KWARGS.xint = 1
_C.ADA.KWARGS.scale = 1
_C.ADA.KWARGS.rotate = 1
_C.ADA.KWARGS.aniso = 1
_C.ADA.KWARGS.xfrac = 1
_C.ADA.KWARGS.brightness = 1
_C.ADA.KWARGS.contrast = 1
_C.ADA.KWARGS.lumaflip = 1
_C.ADA.KWARGS.hue = 1
_C.ADA.KWARGS.saturation = 1

# ------ evaluation ------
_C.EVAL = CN()
_C.EVAL.metrics = ""
_C.EVAL.batch_gpu = 16
_C.EVAL.FID = CN()
_C.EVAL.FID.every = 0
_C.EVAL.FID.batch_gpu = 32
_C.EVAL.FID.n_sample = 50000
_C.EVAL.FID.inception_cache = ""
_C.EVAL.KID = CN()
_C.EVAL.KID.every = 0
_C.EVAL.KID.batch_gpu = 32
_C.EVAL.KID.n_sample = 50000
_C.EVAL.KID.inception_cache = ""


def get_cfg_defaults():
    cfg = _C.clone()
    return cfg


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node

    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
        cfg_dict[k] = convert_to_dict(v, key_list + [k])
    return cfg_dict


def override(cfg: CN, item: dict, copy: bool = False) -> CN:
    "only support 1 level override for simplicity"
    if copy:
        cfg = cfg.clone()

    cfg.defrost()
    for key, override_val in item.items():
        setattr(cfg, key, override_val)
    cfg.freeze()

    if copy:
        return cfg


def configurable(init_func=None, *, from_config=None):
    """
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to arguments.
    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a: cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite
    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """

    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (CN, DictConfig)):
        return True
    if isinstance(kwargs.pop("cfg", None), (CN, DictConfig)):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False
