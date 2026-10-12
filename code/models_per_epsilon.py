import torch
from torch.nn import Sequential
import argparse

from nnpf.problems import AllenCahnProblem as Problem
from nnpf.models import Reaction, HeatArray
from nnpf.utils import get_default_args
from nnpf.nn import Parallel, LinearChannels, Residual
# from nnpf.problems import MCSphereDataset, MCSphereLazyDataset

class DR1(Problem):
    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 train_u=None, train_eps=None,
                 valid_u=None, valid_eps=None,
                 mmap=False, force_float32=False, version=None,
                 **kwargs):
        super().__init__(**kwargs)

        # Fix kernel size to match domain dimension
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters(
            'kernel_size', 'kernel_init',
            'layers', 'activation',
            'train_u', 'train_eps',
            'valid_u', 'valid_eps',
            'mmap', 'force_float32', 'version',
        )

        # Helper function
        def create_diffusion():
            return HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            )

        def create_reaction():
            return Reaction(layers, activation)

        self.model = Sequential(create_diffusion(), create_reaction())

    def training_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data)
            loss += self.loss(data, target)
        loss /= len(targets)
        return self.dispatch_metrics({'loss': loss})

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        for target in targets:
            data = self(data)
            loss += self.loss(data, target)
        loss /= len(targets)
        return {'val_loss': loss}

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("DR1", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--layers', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, help='Name of the activation function')
        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None, help='train u npy path (E,S,N,1,H,W)')
        data.add_argument('--train_eps', type=str, default=None, help='train eps npy path')
        data.add_argument('--valid_u', type=str, default=None, help='valid u npy path (E,S,N,1,H,W)')
        data.add_argument('--valid_eps', type=str, default=None, help='valid eps npy path')
        data.add_argument('--mmap', action='store_true', help="np.load(..., mmap_mode='r')")
        data.add_argument('--force_float32', action='store_true', help="force u_seq float32")
        # =======================================
        group.set_defaults(**{**get_default_args(DR1), **defaults})
        return parser


class DR2(Problem):
    def __init__(self,
                 kernel_size=17, kernel_init='zeros',
                 layers=[8, 3], activation='GaussActivation',
                 train_u=None, train_eps=None,
                 valid_u=None, valid_eps=None,
                 mmap=False, force_float32=False, version=None,
                 **kwargs):
        super().__init__(**kwargs)

        # Fix kernel size to match domain dimension
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]
        else:
            kernel_size = list(kernel_size)
        if len(kernel_size) == 1:
            kernel_size = kernel_size * self.domain.dim

        # Hyper-parameters (used for saving/loading the module)
        self.save_hyperparameters(
            'kernel_size', 'kernel_init',
            'layers', 'activation',
            'train_u', 'train_eps',
            'valid_u', 'valid_eps',
            'mmap', 'force_float32','version',
        )

        # Helper function
        def create_diffusion():
            return HeatArray(
                kernel_size=kernel_size, init=kernel_init,
                bounds=self.hparams.bounds, N=self.hparams.N
            )

        def create_reaction():
            return Reaction(layers, activation)

        # Creating model
        self.model = Sequential(
            LinearChannels(1, 2, bias=False),
            Parallel(
                Sequential(create_diffusion(), create_reaction()),
                Residual(create_reaction()),
            ),
            LinearChannels(2, 1, bias=True),
            create_diffusion(),
            create_reaction(),
        )
    def training_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        # pdb.set_trace()
        for target in targets:
            # data = self(data, epsilon=epsilon)
            data = self(data)
            loss += self.loss(data, target)
        loss /= len(targets)
        return self.dispatch_metrics({'loss': loss})
    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        data, epsilon, *targets = batch
        loss = data.new_zeros([])
        # pdb.set_trace()
        for target in targets:
            data = self(data)
            loss += self.loss(data, target)
        loss /= len(targets)
        # self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'val_loss': loss}

    @staticmethod
    def add_model_specific_args(parent_parser, defaults={}):
        parser = Problem.add_model_specific_args(parent_parser, defaults)
        group = parser.add_argument_group("DR2", "Options specific to this model")
        group.add_argument('--kernel_size', type=int, nargs='+', help='Size of the kernel (nD)')
        group.add_argument('--kernel_init', choices=['zeros', 'random'], help="Initialization of the convolution kernel")
        group.add_argument('--layers', type=int, nargs='+', help='Sizes of the hidden layers')
        group.add_argument('--activation', type=str, help='Name of the activation function')
        data = parser.add_argument_group("Data", "Dataset paths")
        data.add_argument('--train_u', type=str, default=None, help='train u npy path (E,S,N,1,H,W)')
        data.add_argument('--train_eps', type=str, default=None, help='train eps npy path')
        data.add_argument('--valid_u', type=str, default=None, help='valid u npy path (E,S,N,1,H,W)')
        data.add_argument('--valid_eps', type=str, default=None, help='valid eps npy path')
        data.add_argument('--mmap', action='store_true', help="np.load(..., mmap_mode='r')")
        data.add_argument('--force_float32', action='store_true', help="force u_seq float32")
        # =======================================
        group.set_defaults(**{**get_default_args(DR2), **defaults})
        return parser
