#!/usr/bin/env python3
import argparse
import os.path
import numpy as np
import torch
from torch import optim
from torch import nn
import utils
from data import get_dataset, DATASET_CONFIGS
from train import train
from dgr import Scholar
from models import WGAN, CNN


parser = argparse.ArgumentParser('./main.py', description='PyTorch implementation: Deep Generative Replay')
main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train', action='store_true', help='optimize new model')
main_command.add_argument('--test', action='store_false', dest='train', help='use existing model to generate examples')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")

# expirimental task parameters.
task_params = parser.add_argument_group('Task Parameters', 'set experimental tasks')
task_params.add_argument('--experiment', type=str, choices=['permMNIST', 'svhn-mnist', 'mnist-svhn'],
                         default='permMNIST')
task_params.add_argument('--n-tasks', type=int, default=5, help='number of permutation for permMNIST-task')

# model architecture parameters.
model_params = parser.add_argument_group('Model Parameters', 'define model architecture')
# -solver
model_params.add_argument('--solver-depth', type=int, default=5)
model_params.add_argument('--solver-reducing-layers', type=int, default=3)
model_params.add_argument('--solver-channel-size', type=int, default=1024)
# -replay / generator
model_params.add_argument('--replay-mode', type=str, default='generative-replay',
                          choices=['exact-replay', 'generative-replay', 'none'])
model_params.add_argument('--generator-z-size', type=int, default=100)
model_params.add_argument('--generator-c-channel-size', type=int, default=64)
model_params.add_argument('--generator-g-channel-size', type=int, default=64)

# training hyperparameters.
train_params = parser.add_argument_group('Training Parameters', 'hyper-parameters for training')
train_params.add_argument('--lamda', type=float, default=10.,
                          help="how strong to weight the generator 'gradient penalty'")
train_params.add_argument('--critic-updates', type=int, default=5, dest='cu',
                          help="# steps (per batch) to optimize generator's critic")
train_params.add_argument('--gen-iter', type=int, default=3000, help="# batches to optimize generator")
train_params.add_argument('--sol-iter', type=int, default=1000, help="# batches to optimize solver")
train_params.add_argument('--rnt', type=float, default=.3, help="importance of new task")
train_params.add_argument('--lr', type=float, default=1e-04, help="learning rate")
train_params.add_argument('--beta1', type=float, default=0.5, help="parameter of adam-optimizer")
train_params.add_argument('--beta2', type=float, default=0.9, help="parameter of adam-optimizer")
train_params.add_argument('--decay', type=float, default=1e-05, help="weight-decay")
train_params.add_argument('--batch', type=int, default=32, help="batch-size")

# evaluation parameters.
eval_params = parser.add_argument_group('Evaluation Parameters', 'evaluate model performance')
eval_params.add_argument('--test-N', type=int, default=1024, dest="test_n",
                         help="(plotting) # samples for evaluating solver's precision")
eval_params.add_argument('--eval-log', type=int, default=50,
                         help="(plotting) # iters after which to evaluate precision")
eval_params.add_argument('--loss-log', type=int, default=30, help="(plotting) # iters after which to plot loss")
eval_params.add_argument('--samples', action='store_true', help="(samples) plot generated images")
eval_params.add_argument('--sample-size', type=int, default=36, dest='s_size',
                         help="(samples) # generated images to sample")
eval_params.add_argument('--sample-log', type=int, default=200, dest='s_log',
                         help="(samples) # iters after which to generate images")
eval_params.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir')
eval_params.add_argument('--checkpoint-dir', type=str, default='./checkpoints', dest='c_dir')
eval_params.add_argument('--sample-dir', type=str, default='./samples', dest='s_dir')


# which proportion of the training-set should be used as validation set?
valid_proportion = 1./6

if __name__ == '__main__':
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda

    # set random seeds.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # prepare data for chosen experiment.
    capacity = np.ceil(args.batch * max(args.gen_iter, args.sol_iter) / (1-valid_proportion))
    if args.experiment == 'permMNIST':
        # data-set configurations to use.
        config = DATASET_CONFIGS['mnist']
        # generate permutations.
        permutations = [np.random.permutation(config['size']**2) for _ in range(args.n_tasks)]
        # prepare datasets.
        train_datasets = [get_dataset(
            'mnist', permutation=p, capacity=capacity, data_dir=args.d_dir
        ) for p in permutations]
        test_datasets = [get_dataset(
            'mnist', train=False, permutation=p, capacity=capacity, data_dir=args.d_dir
        ) for p in permutations]
    elif args.experiment in ('svhn-mnist', 'mnist-svhn'):
        # data-set configurations to use.
        config = DATASET_CONFIGS['mnist-color']
        # prepare individual datasets.
        mnist_color_train = get_dataset('mnist-color', train=True, capacity=capacity, data_dir=args.d_dir)
        mnist_color_test = get_dataset('mnist-color', train=False, capacity=capacity, data_dir=args.d_dir)
        svhn_train = get_dataset('svhn', train=True, capacity=capacity, data_dir=args.d_dir)
        svhn_test = get_dataset('svhn', train=False, capacity=capacity, data_dir=args.d_dir)
        # combine the datasets.
        if args.experiment == 'mnist-svhn':
            train_datasets = [mnist_color_train, svhn_train]
            test_datasets = [mnist_color_test, svhn_test]
        else:
            train_datasets = [svhn_train, mnist_color_train]
            test_datasets = [svhn_test, mnist_color_test]
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(args.experiment))

    # define the models.
    cnn = CNN(
        image_size=config['size'], image_channel_size=config['channels'], classes=config['classes'],
        depth=args.solver_depth, channel_size=args.solver_channel_size, reducing_layers=args.solver_reducing_layers,
    )
    wgan = WGAN(
        image_size=config['size'], image_channel_size=config['channels'], z_size=args.generator_z_size,
        c_channel_size=args.generator_c_channel_size, g_channel_size=args.generator_g_channel_size,
    )
    label = '{experiment}-{replay_mode}-r{rnt}'.format(
        experiment=args.experiment, replay_mode=args.replay_mode,
        rnt=1 if args.replay_mode == 'none' else args.rnt,
    )
    scholar = Scholar(label, generator=wgan, solver=cnn)

    # initialize the model.
    utils.gaussian_intiailize(scholar, std=.02)

    # use cuda if needed
    if cuda:
        scholar.cuda()

    # determine whether we need to train the generator or not.
    train_generator = (args.replay_mode == 'generative-replay' or args.samples)

    # define & set criterion and optimizer for the scholar's solver.
    solver_criterion = nn.CrossEntropyLoss()
    solver_optimizer = optim.Adam(scholar.solver.parameters(),
                                  lr=args.lr, weight_decay=args.decay, betas=(args.beta1, args.beta2))
    scholar.solver.set_criterion(solver_criterion)
    scholar.solver.set_optimizer(solver_optimizer)

    # define & set criterion and optimizer for the scholar's generator.
    generator_g_optimizer = optim.Adam(scholar.generator.generator.parameters(),
                                       lr=args.lr, weight_decay=args.decay, betas=(args.beta1, args.beta2))
    generator_c_optimizer = optim.Adam(scholar.generator.critic.parameters(),
                                       lr=args.lr, weight_decay=args.decay, betas=(args.beta1, args.beta2))
    scholar.generator.set_generator_optimizer(generator_g_optimizer)
    scholar.generator.set_critic_optimizer(generator_c_optimizer)
    # set additional settings for the scholar's generator
    scholar.generator.set_lambda(args.lamda)
    scholar.generator.set_critic_updates_per_batch(args.cu)

    # run the experiment.
    if args.train:
        train(
            scholar, train_datasets, test_datasets,
            replay_mode=args.replay_mode,
            generator_iterations=args.gen_iter if train_generator else 0,
            solver_iterations=args.sol_iter,
            importance_of_new_task=args.rnt,
            batch_size=args.batch,
            test_size=args.test_n,
            sample_size=args.s_size,
            loss_log_interval=args.loss_log,
            eval_log_interval=args.eval_log,
            sample_log_interval=args.s_log,
            sample_log=args.samples,
            sample_dir=args.s_dir,
            checkpoint_dir=args.c_dir,
            collate_fn=utils.label_squeezing_collate_fn,
            cuda=cuda,
            valid_proportion=valid_proportion
        )
    else:
        path = os.path.join(args.s_dir, '{}-sample'.format(scholar.name))
        utils.load_checkpoint(scholar, args.c_dir)
        utils.test_model(scholar.generator, args.s_size, path)
