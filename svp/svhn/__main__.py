from typing import Tuple, Optional

import torch
import click
import torch.backends.cudnn as cudnn

from svp.common import utils
from svp.common.cli import computing_options, miscellaneous_options
from svp.common.active import SELECTION_METHODS as active_learning_methods
from svp.common.coreset import SELECTION_METHODS as coreset_methods
from svp.svhn.models import MODELS
from svp.svhn.datasets import DATASETS
from svp.svhn.train import train as train_function
from svp.svhn.active import active as active_function
from svp.svhn.coreset import coreset as coreset_function


@click.group()
def cli():
    pass


def dataset_options(func):
    # Dataset options
    decorators = [
        click.option('--datasets-dir', default='./data', show_default=True,
                     help='Path to datasets.'),
        click.option('--dataset', '-d', type=click.Choice(DATASETS),
                     default='svhn', show_default=True,
                     help='Specify dataset to use in experiment.'),
        click.option('--dataset-size', type=int,
                     default=73257, show_default=True,
                     help='Specify dataset size'),
        click.option('--augmentation/--no-augmentation',
                     default=True, show_default=True,
                     help='Add data augmentation.'),
        click.option('--validation', '-v', default=0, show_default=True,
                     help='Number of examples to use for valdiation'),
        click.option('--shuffle/--no-shuffle', default=False, show_default=True,
                     help=('Shuffle train and validation data before'
                           ' splitting.'))
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func


def training_options(func):
    decorators = [
        click.option('--arch', '-a', type=click.Choice(MODELS.keys()),
                     default='preact20', show_default=True,
                     help='Specify model architecture.'),
        click.option('--optimizer', '-o', type=click.Choice(['sgd', 'adam']),
                     default='sgd', show_default=True,
                     help='Specify optimizer for training.'),
        click.option('--epochs', '-e', multiple=True, type=int,
                     default=(1, 90, 45, 45), show_default=True,
                     help='Specify epochs for training.'),
        click.option('learning_rates', '--learning-rate', '-l', multiple=True,
                     type=float, default=(0.01, 0.1, 0.01, 0.001),
                     show_default=True,
                     help='Specify learning rate for training.'),
        click.option('--momentum', type=float, default=0.9, show_default=True,
                     help='Specify momentum.'),
        click.option('--weight-decay', type=float, default=5e-4,
                     show_default=True,
                     help='Specify weight decay.'),
        click.option('--batch-size', '-b', default=128, show_default=True,
                     help='Specify minibatch size for training.'),
        click.option('--eval-batch-size', type=int,
                     callback=utils.override_option,
                     help='Override minibatch size for evaluation')
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func


@cli.command()
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
@dataset_options
@training_options
@computing_options
@miscellaneous_options
def train(run_dir: str,

          datasets_dir: str, dataset: str, augmentation: bool,
          validation: int, shuffle: bool,

          arch: str, optimizer: str, epochs: Tuple[int, ...],
          learning_rates: Tuple[float, ...],
          momentum: float, weight_decay: float,
          batch_size: int, eval_batch_size: int,

          cuda: bool, device_ids: Tuple[int, ...],
          num_workers: int, eval_num_workers: int,

          seed: int, checkpoint: str, track_test_acc: bool):
    train_function(**locals())


def proxy_training_overrides(func):
    decorators = [
        click.option('--proxy-arch', '-a', type=click.Choice(MODELS.keys()),
                     callback=utils.override_option,
                     help='Override proxy model architecture.'),
        click.option('--proxy-optimizer', '-o',
                     type=click.Choice(['sgd', 'adam']),
                     callback=utils.override_option,
                     help='Specify optimizer for training.'),
        click.option('--proxy-epochs', multiple=True, type=int,
                     callback=utils.override_option,
                     help='Override epochs for proxy training.'),
        click.option('proxy_learning_rates', '--proxy-learning-rate',
                     multiple=True, type=float,
                     callback=utils.override_option,
                     help='Override proxy learning rate for training.'),
        click.option('--proxy-momentum', type=float,
                     callback=utils.override_option,
                     help='Override momentum.'),
        click.option('--proxy-weight-decay', type=float,
                     callback=utils.override_option,
                     help='Override weight decay.'),
        click.option('--proxy-batch-size', type=int,
                     callback=utils.override_option,
                     help='Override proxy minibatch size for training.'),
        click.option('--proxy-eval-batch-size', type=int,
                     callback=utils.override_option,
                     help='Override proxy minibatch size for evaluation'),
    ]
    decorators.reverse()
    for decorator in decorators:
        func = decorator(func)
    return func


@cli.command()
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
@dataset_options
@training_options
@proxy_training_overrides
# Active learning options
@click.option('--initial-subset', type=int, default=1_000, show_default=True,
              help='Number of randomly selected training examples to use for'
                   ' initial labeled set.')
@click.option('rounds', '--round', '-r', multiple=True, type=int,
              default=(), show_default=True,
              help='Number of unlabeled examples to select in a round of'
                   ' labeling.')
@click.option('--selection-method', type=click.Choice(active_learning_methods),
              default='least_confidence', show_default=True,
              help='Criteria for selecting unlabeled examples to label')
@click.option('--precomputed-selection',
              help='Path to timestamped run_dir of precomputed indices')
@click.option('--train-target/--no-train-target',
              default=True, show_default=True,
              help=('If proxy and target are different, train the target'
                    ' after each round of selection'))
@click.option('--eval-target-at', multiple=True, type=int,
              help=('If proxy and target are different and --train-target,'
                    ' limit the evaluation of the target model to specific'
                    ' labeled subset sizes'))
@click.option('--initial-balance', type=bool, default=False, show_default=False,
              help=('Binary varialbe that turn on/off initial balance'))
@click.option('--enable-intersect', type=bool, default=False, show_default=False,
              help=('Enable intersection between coreset and active'))
@click.option('--weighted-loss', type=bool, default=False, show_default=False,
              help=('Enable per-sample weighted loss'))
@click.option('--intersect-method', type=str, default='naive', show_default=False,
              help=('Choose intersection methods'))
@click.option('--cross-test', type=str, default='none', show_default=False,
              help=('Choose cross test'))
@click.option('--intersect-rate', type=float, default=0.5, show_default=False,
              help=('The mixing rate of coreset, ranging from 0 to 1')) #
#@click.option('--weighted-loss', type=bool, default=False, show_default=False,
#              help=('Binary varialbe that turn on/off weighted per-sample loss'))
@click.option('--initial-num-per-class', type=int, default=1_00, show_default=True,
              help='The number of initial subset seize for each class')
@click.option('--subset-bias', type=int, default=0, show_default=True,
              help='Offset of subset of entire dataset')
@click.option('--runs', type=int, default=0, show_default=True,
              help='runnin number')
@click.option('--coreset-path',
              help='Path to coreset')
@click.option('--coreset-loss-path',
              help='Path to coreset loss')

@computing_options
@miscellaneous_options
def active(run_dir: str,

           datasets_dir: str, dataset: str, dataset_size:int,
           augmentation: bool, validation: int, shuffle: bool, initial_balance : bool,

           subset_bias: int,
           initial_num_per_class: int, 
           coreset_path: str, 
           coreset_loss_path: str, 
           weighted_loss: bool,
           enable_intersect: bool,
           intersect_method: str,
           intersect_rate: float, 
           runs: int, 
           cross_test: str,
           
           arch: str, optimizer: str,
           epochs: Tuple[int, ...],
           learning_rates: Tuple[float, ...],
           momentum: float, weight_decay: float,
           batch_size: int, eval_batch_size: int,

           proxy_arch: str, proxy_optimizer: str,
           proxy_epochs: Tuple[int, ...],
           proxy_learning_rates: Tuple[float, ...],
           proxy_momentum: float, proxy_weight_decay: float,
           proxy_batch_size: int, proxy_eval_batch_size: int,

           initial_subset: int,  rounds: Tuple[int, ...],
           selection_method: str, precomputed_selection: Optional[str],
           train_target: bool, eval_target_at: Tuple[int, ...],

           cuda: bool, device_ids: Tuple[int, ...],
           num_workers: int, eval_num_workers: int,

           seed: int, checkpoint: str, track_test_acc: bool):
    active_function(**locals())

@cli.command()
@click.option('--run-dir', default='./run', show_default=True,
              help='Path to log results and other artifacts.')
@dataset_options
@training_options
@proxy_training_overrides
# Core-set selection options
@click.option('--subset', type=int, default=10_000, show_default=True,
              help='Number of examples to keep in the selected subset.')
@click.option('--selection-method', type=click.Choice(coreset_methods),
              default='least_confidence', show_default=True,
              help='Criteria for selecting examples')
@click.option('--precomputed-selection',
              help='Path to timestamp run_dir of precomputed indices')
@click.option('--train-target/--no-train-target',
              default=True, show_default=True,
              help=('If proxy and target are different, train the target'
                    ' after selection'))
@click.option('--runs', type=int, default=0, show_default=True,
              help='running number')
@computing_options
@miscellaneous_options
def coreset(run_dir: str,

            datasets_dir: str, dataset: str, augmentation: bool,
            validation: int, shuffle: bool, dataset_size: int,
            runs: int, 
            arch: str, optimizer: str,
            epochs: Tuple[int, ...],
            learning_rates: Tuple[float, ...],
            momentum: float, weight_decay: float,
            batch_size: int, eval_batch_size: int,

            proxy_arch: str, proxy_optimizer: str,
            proxy_epochs: Tuple[int, ...],
            proxy_learning_rates: Tuple[float, ...],
            proxy_momentum: float, proxy_weight_decay: float,
            proxy_batch_size: int, proxy_eval_batch_size: int,

            subset: int, selection_method: str,
            precomputed_selection: Optional[str], train_target: bool,

            cuda: bool, device_ids: Tuple[int, ...],
            num_workers: int, eval_num_workers: int,

            seed: int, checkpoint: str, track_test_acc: bool):
    coreset_function(**locals())


if __name__ == '__main__':
    if torch.cuda.is_available():
        cudnn.benchmark = True  # type: ignore
    cli()
