import os
from glob import glob
from typing import Tuple, Optional
from functools import partial

import numpy as np
from torch import cuda

from svp.common import utils
from svp.common.train import create_loaders
from svp.svhn.datasets import create_dataset
from svp.svhn.train import create_model_and_optimizer
from svp.common.selection import select
from svp.common.active import (generate_models,
                               check_different_models,
                               symlink_target_to_proxy,
                               symlink_to_precomputed_proxy,
                               validate_splits)


def active(run_dir: str = './run',

           datasets_dir: str = './data', dataset: str = 'svhn', dataset_size: int=73257,
           augmentation: bool = True, validation: int = 0, shuffle: bool = False,
           initial_balance : bool = False,
           initial_num_per_class: int = 100,

           subset_bias: int=0,
           coreset_path: str=None,
           coreset_loss_path: str=None,
		   weighted_loss: bool = False,
           enable_intersect: bool = False,
		   intersect_method: str ='naive',
           intersect_rate: float=0.5, 
           runs: int=0,

           arch: str = 'preact20', optimizer: str = 'sgd',
           epochs: Tuple[int, ...] = (1, 90, 45, 45),
           learning_rates: Tuple[float, ...] = (0.01, 0.1, 0.01, 0.001),
           momentum: float = 0.9, weight_decay: float = 5e-4,
           batch_size: int = 128, eval_batch_size: int = 128,

           proxy_arch: str = 'preact20', proxy_optimizer: str = 'sgd',
           proxy_epochs: Tuple[int, ...] = (1, 90, 45, 45),
           proxy_learning_rates: Tuple[float, ...] = (0.01, 0.1, 0.01, 0.001),
           proxy_momentum: float = 0.9, proxy_weight_decay: float = 5e-4,
           proxy_batch_size: int = 128, proxy_eval_batch_size: int = 128,

           initial_subset: int = 1_000,
           rounds: Tuple[int, ...] = (4_000, 5_000, 5_000, 5_000, 5_000),
           selection_method: str = 'least_confidence',
           precomputed_selection: Optional[str] = None,
           train_target: bool = True,
           eval_target_at: Optional[Tuple[int, ...]] = None,

           cuda: bool = True,
           device_ids: Tuple[int, ...] = tuple(range(cuda.device_count())),
           num_workers: int = 0, eval_num_workers: int = 0,

           seed: Optional[int] = None, checkpoint: str = 'best',
           track_test_acc: bool = True):
    """
    Perform active learning on CIFAR10 and CIFAR100.

    If the model architectures (`arch` vs `proxy_arch`) or the learning rate
    schedules don't match, "selection via proxy" (SVP) is performed and two
    separate models are trained. The proxy is used for selecting which
    examples to label, while the target is only used for evaluating the
    quality of the selection. By default, the target model (`arch`) is
    trained and evaluated after each selection round. To change this behavior
    set `eval_target_at` to evaluate at a specific labeling budget(s) or set
    `train_target` to False to skip evaluating the target model. You can
    evaluate a series of selections later using the `precomputed_selection`
    option.

    Parameters
    ----------
    run_dir : str, default './run'
        Path to log results and other artifacts.
    datasets_dir : str, default './data'
        Path to datasets.
    dataset : str, default 'cifar10'
        Dataset to use in experiment (i.e., CIFAR10 or CIFAR100)
    augmentation : bool, default True
        Add data augmentation (i.e., random crop and horizontal flip).
    validation : int, default 0
        Number of examples from training set to use for valdiation.
    shuffle : bool, default True
        Shuffle training data before splitting into training and validation.
    arch : str, default 'preact20'
        Model architecture for the target model. `preact20` is short for
        ResNet20 w/ Pre-Activation.
    optimizer : str, default = 'sgd'
        Optimizer for training the target model.
    epochs : Tuple[int, ...], default (1, 90, 45, 45)
        Epochs for training the target model. Each number corresponds to a
        learning rate below.
    learning_rates : Tuple[float, ...], default (0.01, 0.1, 0.01, 0.001)
        Learning rates for training the target model. Each learning rate is
        used for the corresponding number of epochs above.
    momentum : float, default 0.9
        Momentum for SGD with the target model.
    weight_decay : float, default 5e-4
        Weight decay for SGD with the target model.
    batch_size : int, default 128
        Minibatch size for training the target model.
    eval_batch_size : int, default 128
        Minibatch size for evaluation (validation and testing) of the target
        model.
    proxy_arch : str, default 'preact20'
        Model architecture for the proxy model. `preact20` is short for
        ResNet20 w/ Pre-Activation.
    proxy_optimizer : str, default = 'sgd'
        Optimizer for training the proxy model.
    proxy_epochs : Tuple[int, ...], default (1, 90, 45, 45)
        Epochs for training the proxy model. Each number corresponds to a
        learning rate below.
    proxy_learning_rates : Tuple[float, ...], default (0.01, 0.1, 0.01, 0.001)
        Learning rates for training the proxy model. Each learning rate is
        used for the corresponding number of epochs above.
    proxy_momentum : float, default 0.9
        Momentum for SGD with the proxy model.
    proxy_weight_decay : float, default 5e-4
        Weight decay for SGD with the proxy model.
    proxy_batch_size : int, default 128
        Minibatch size for training the proxy model.
    proxy_eval_batch_size : int, default 128
        Minibatch size for evaluation (validation and testing) of the model
        proxy.
    initial_subset : int, default 1,000
        Number of randomly selected training examples to use for the initial
        labeled set.
    rounds : Tuple[int, ...], default (4,000, 5,000, 5,000, 5,000, 5,000)
        Number of unlabeled exampels to select in a round of labeling.
    selection_method : str, default least_confidence
        Criteria for selecting unlabeled examples to label.
    precomputed_selection : str or None, default None
        Path to timestamped run_dir of precomputed indices.
    train_target : bool, default True
        If proxy and target are different, train the target after each round
        of selection or specific rounds as specified below.
    eval_target_at : Tuple[int, ...] or None, default None
        If proxy and target are different and `train_target`, limit the
        evaluation of the target model to specific labeled subset sizes.
    cuda : bool, default True
        Enable or disable use of available GPUs
    device_ids : Tuple[int, ...], default True
        GPU device ids to use.
    num_workers : int, default 0
        Number of data loading workers for training.
    eval_num_workers : int, default 0
        Number of data loading workers for evaluation.
    seed : Optional[int], default None
        Random seed for numpy, torch, and others. If None, a random int is
        chosen and logged in the experiments config file.
    checkpoint : str, default 'best'
        Specify when to create a checkpoint for the model: only checkpoint the
        best performing model on the validation data or the training data if
        `validation == 0` ("best"), after every epoch ("all"), or only the last
        epoch of each segment of the learning rate schedule ("last").
    track_test_acc : bool, default True
        Calculate performance of the models on the test data in addition or
        instead of the validation dataset.'
    """
    # Set seeds for reproducibility.
    seed = utils.set_random_seed(seed)
    # Capture all of the arguments to save alongside the results.
    config = utils.capture_config(**locals())
    # Create a unique timestamped directory for this experiment.
    run_dir = utils.create_run_dir(run_dir, timestamp=config['timestamp'])
    utils.save_config(config, run_dir)
    # Update the computing arguments based on the runtime system.
    use_cuda, device, device_ids, num_workers = utils.config_run_env(
            cuda=cuda, device_ids=device_ids, num_workers=num_workers)

    # Create the training dataset.
    train_dataset = create_dataset(dataset, datasets_dir, train=True,
                                   augmentation=augmentation)
    # Verify there is enough training data for validation,
    #   the initial subset, and the selection rounds.
    validate_splits(train_dataset, validation, initial_subset, rounds)

    # Create the test dataset.
    test_dataset = None
    if track_test_acc:
        test_dataset = create_dataset(dataset, datasets_dir, train=False,
                                      augmentation=False)

    # Calculate the number of classes (e.g., 10 or 100) so the model has
    #   the right dimension for its output.
    num_classes = len(set(train_dataset.labels))  # type: ignore

    # Split the training dataset between training and validation.
    unlabeled_pool, dev_indices = utils.split_indices(
        train_dataset, validation, run_dir, shuffle=shuffle)
    # Create the proxy to select which data points to label. If the
    #   selections were precomputed in another run or elsewhere, we can
    #   ignore this step.
    if precomputed_selection is None:
        # Use a partial so the appropriate model can be created without
        #   arguments.
        proxy_partial = partial(create_model_and_optimizer,
                                arch=proxy_arch,
                                num_classes=num_classes,
                                optimizer=proxy_optimizer,
                                learning_rate=proxy_learning_rates[0],
                                momentum=proxy_momentum,
                                weight_decay=proxy_weight_decay)

        # Create a directory for the proxy results to avoid confusion.
        proxy_run_dir = os.path.join(run_dir, 'proxy')
        os.makedirs(proxy_run_dir, exist_ok=True)
        # Create data loaders for validation and testing. The training
        #   data loader changes as labeled data is added, so it is
        #   instead a part of the proxy model generator below.
        _, proxy_dev_loader, proxy_test_loader = create_loaders(
            train_dataset,
            batch_size=proxy_batch_size,
            eval_batch_size=proxy_eval_batch_size,
            test_dataset=test_dataset,
            use_cuda=use_cuda,
            num_workers=num_workers,
            eval_num_workers=eval_num_workers,
            indices=(unlabeled_pool, dev_indices))

        # Create the proxy model generator (i.e., send data and get a
        #   trained model).
        proxy_generator = generate_models(
            proxy_partial, proxy_epochs, proxy_learning_rates,
            train_dataset,  proxy_batch_size,
            device, use_cuda,
            num_workers=num_workers,
            device_ids=device_ids,
            dev_loader=proxy_dev_loader,
            test_loader=proxy_test_loader,
            run_dir=proxy_run_dir,
            checkpoint=checkpoint)
        # Start the generator
        next(proxy_generator)

    # Check that the proxy and target are different models
    are_different_models = check_different_models(config)
    # Maybe create the target.
    if train_target:
        # If the proxy and target models aren't different, we don't
        #   need to create a separate model generator*.
        # * Unless the proxy wasn't created because the selections were
        #   precomputed (see above).
        if are_different_models or precomputed_selection is not None:
            # Use a partial so the appropriate model can be created
            #   without arguments.
            target_partial = partial(create_model_and_optimizer,
                                     arch=arch,
                                     num_classes=num_classes,
                                     optimizer=optimizer,
                                     learning_rate=learning_rates[0],
                                     momentum=momentum,
                                     weight_decay=weight_decay)

            # Create a directory for the target to avoid confusion.
            target_run_dir = os.path.join(run_dir, 'target')
            os.makedirs(target_run_dir, exist_ok=True)
            # Create data loaders for validation and testing. The training
            #   data loader changes as labeled data is added, so it is
            #   instead a part of the target model generator below.
            _, target_dev_loader, target_test_loader = create_loaders(
                train_dataset,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                test_dataset=test_dataset,
                use_cuda=use_cuda,
                num_workers=num_workers,
                eval_num_workers=eval_num_workers,
                indices=(unlabeled_pool, dev_indices))

            # Create the target model generator (i.e., send data and
            #   get a trained model).
            target_generator = generate_models(
                target_partial, epochs, learning_rates,
                train_dataset,  batch_size,
                device, use_cuda,
                num_workers=num_workers,
                device_ids=device_ids,
                dev_loader=target_dev_loader,
                test_loader=target_test_loader,
                run_dir=target_run_dir,
                checkpoint=checkpoint)
            # Start the generator
            next(target_generator)
        else:
            # Proxy and target are the same, so we can just symlink
            symlink_target_to_proxy(run_dir)

    # Perform active learning.
    if precomputed_selection is not None:
        assert train_target, "Must train target if selection is precomuted"
        assert os.path.exists(precomputed_selection)

        # Collect the files with the previously selected data.
        files = glob(os.path.join(precomputed_selection, 'proxy',
                     '*', 'labeled_*.index'))
        indices = [np.loadtxt(file, dtype=np.int64) for file in files]
        # Sort selections by length to replicate the order data was
        #   labeled.
        selections = sorted(
            zip(files, indices),
            key=lambda selection: len(selection[1]))  # type: ignore

        # Symlink proxy directories and files for convenience.
        symlink_to_precomputed_proxy(precomputed_selection, run_dir)

        # Train the target model on each selection.
        for file, labeled in selections:
            print('Load labeled indices from {}'.format(file))
            # Check whether the target model should be trained. If you
            #   have a specific labeling budget, you may not want to
            #   evaluate the target after each selection round to save
            #   time.
            should_eval = (eval_target_at is None or
                           len(eval_target_at) == 0 or
                           len(labeled) in eval_target_at)
            if should_eval:
                # Train the target model on the selected data.
                _, stats = target_generator.send(labeled)
                utils.save_result(stats, os.path.join(run_dir, "target.csv"))
    else:  # Select which points to label using the proxy.
        # Create initial random subset to train the proxy (warm start).
        labeled = []
        sim_loss = np.array([])
        sim_wloss = np.array([])
        core_indices = np.array([])
        half_core_indices = np.array([]) # half core indices
        isCoreLoss = False

        if coreset_loss_path is not None:
            isCoreLoss = True # Core Loss is/isn't defined

        if coreset_path is not None:
            isCoreset = True # Core Set is/isn't defined
            core_indices = np.loadtxt(coreset_path,dtype=int) # Load core Indices resulting from self-supervised learning
            												  # size : 50,000
            if isCoreLoss:
                sim_loss = np.loadtxt(coreset_loss_path,dtype=float) # Load core(simmmiarity) Loss resulting 
                													 # from self-supervised learning
                													 # size : 50,000
                norm_sim_loss = (2000-sim_loss)/2000 # extract cossim from simloss
                ### normalize scale between 0 and 1 ###
                sim_min = np.min(sim_loss)
                sim_max = np.max(sim_loss)
                sim_loss = (sim_loss-sim_min)/(sim_max-sim_min)
                ### normalize mean and std ####
                
                core_loss = np.zeros(dataset_size) # subset core loss in CIFAR-10 indices order
                sim_wloss = np.zeros(dataset_size)
                norm_sim_wloss = np.zeros(dataset_size)
                sim_wloss[core_indices] = sim_loss # Reorder sim_loss in CIFAR-10 indices order # Map back to original indices
                norm_sim_wloss[core_indices] = norm_sim_loss
            if 'noncore' in coreset_path:
                isCoreset = False

            if initial_balance:
                labeled = utils.split_balanced_core_indices(train_dataset, core_indices, core_loss, isCoreset,
                                                          initial_num_per_class, num_classes)
            else:
                if isCoreset: # core_subset
                    #labeled = core_indices[-initial_subset:]
                    half_core_indices = core_indices[-25000:]
                    labeled = np.argsort(norm_sim_wloss)[subset_bias:subset_bias+initial_subset]
                    if isCoreLoss:
                        core_loss[labeled] = sim_wloss[labeled]
                else : # noncore_subset
                    print('############# non coreset ###############')
                    #labeled = core_indices[:initial_subset]
                    #half_core_indices = core_indices[:25000]
                    #if isCoreLoss:
                    #    core_loss[labeled] = sim_loss[:initial_subset]

            if weighted_loss:
                core_arg = [labeled, core_loss] # For per-sample weighted loss
                print("############ Weighted Per-sample Loss ###############")
            else:
                core_arg = labeled # For uniform loss
                print("############ Uniform Per-sample Loss ###############")

            # Save the index of the initial random subset
            utils.save_index(labeled, run_dir,
                             'initial_subset_{}.index'.format(len(labeled)))
            # Train the proxy on the initial random subset
            model, stats = proxy_generator.send(core_arg)

        else:

            if initial_balance:
                print("initial random subset with balanced data")
                labeled=utils.split_balanced_indices(train_dataset, initial_num_per_class, num_classes)
            else:
                print("initial random subset with no balanced data")
                labeled = np.random.permutation(unlabeled_pool)[:initial_subset]

            # Save the index of the initial random subset
            utils.save_index(labeled, run_dir,
                             'initial_subset_{}.index'.format(len(labeled)))
            # Train the proxy on the initial random subset
            model, stats = proxy_generator.send(labeled)
        
        stats['weighted_loss'] = weighted_loss
        stats['self-supervised'] = coreset_path
        stats['runs'] = runs
        stats['dataset'] = dataset
        stats['initial size'] = len(labeled)
        stats['subset bias'] = subset_bias
        utils.save_result(stats, os.path.join(run_dir, "proxy.csv"))

        for selection_size in rounds:
            # Select additional data to label from the unlabeled pool
            labeled, stats = select(model, train_dataset,
                                    enable_intersect=enable_intersect,
                                    intersect_method=intersect_method,
                                    half_core_indices=half_core_indices,						
                                    sim_loss = sim_wloss,
                                    mixing_rate = intersect_rate,
                                    current=labeled,
                                    pool=unlabeled_pool,
                                    budget=selection_size,
                                    method=selection_method,
                                    batch_size=proxy_eval_batch_size,
                                    device=device,
                                    device_ids=device_ids,
                                    num_workers=num_workers,
                                    use_cuda=use_cuda)
            utils.save_result(stats, os.path.join(run_dir, 'selection.csv'))
            
            # Train the proxy on the newly added data.
            if weighted_loss : # For per-sample weighted loss
                core_loss = np.zeros(dataset_size)
                core_loss[labeled] = sim_wloss[labeled]
                core_arg = [labeled, core_loss]
                model, stats = proxy_generator.send(core_arg)
            else: # For uniform loss
                print("############ Uniform Per-sample Loss ###############")
                model, stats = proxy_generator.send(labeled)             

            utils.save_result(stats, os.path.join(run_dir, 'proxy.csv'))

            # Check whether the target model should be trained. If you
            #   have a specific labeling budget, you may not want to
            #   evaluate the target after each selection round to save
            #   time.
            should_eval = (eval_target_at is None or
                           len(eval_target_at) == 0 or
                           len(labeled) in eval_target_at)
            if train_target and should_eval and are_different_models:
                # Train the target model on the selected data.
                _, stats = target_generator.send(labeled)
                utils.save_result(stats, os.path.join(run_dir, "target.csv"))
