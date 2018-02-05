import abc
import utils
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import ConcatDataset


# ============
# Base Classes
# ============

class GenerativeMixin(object, metaclass=abc.ABCMeta):
    """Abstract base class which requires subclasses to define a sampling interface for a generative model."""
    @abc.abstractmethod
    def sample(self, size):
        pass


class BatchTrainable(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class which requires subclasses to define a generative-replay-based training interface."""
    @abc.abstractmethod
    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        pass



# ==============================
# Deep Generative Replay Modules
# ==============================

class Generator(GenerativeMixin, BatchTrainable):
    """Abstract generator module (for a scholar module).

    Requires subclasses to provide a 'sample' and 'train_a_batch' method."""


class Solver(BatchTrainable):
    """Abstract solver module (for a scholar module).

    Adds a 'solve' and 'train_a_batch' method to subclasses, and requires them to provide a 'forward' method."""
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.criterion = None

    @abc.abstractmethod
    def forward(self, x):
        pass

    def solve(self, x):
        scores = self(x)
        _, predictions = torch.max(scores, 1)
        return predictions

    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=.5):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()

        # clear gradients.
        batch_size = x.size(0)
        self.optimizer.zero_grad()

        # run the model on the real data.
        real_scores = self.forward(x)
        real_loss = self.criterion(real_scores, y)
        _, real_predicted = real_scores.max(1)
        real_prec = (y == real_predicted).sum().data[0] / batch_size

        # run the model on the replayed data.
        if x_ is not None and y_ is not None:
            replay_scores = self.forward(x_)
            replay_loss = self.criterion(replay_scores, y_)
            _, replay_predicted = replay_scores.max(1)
            replay_prec = (y_ == replay_predicted).sum().data[0] / batch_size

            # calculate joint loss of real data and replayed data.
            loss = (
                importance_of_new_task * real_loss +
                (1-importance_of_new_task) * replay_loss
            )
            precision = (real_prec + replay_prec) / 2
        else:
            loss = real_loss
            precision = real_prec

        loss.backward()
        self.optimizer.step()
        return {'loss': loss.data[0], 'precision': precision}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion


class Scholar(GenerativeMixin, nn.Module):
    """Scholar module for Deep Generative Replay."""

    def __init__(self, label, generator, solver):
        '''Instantiate a new Scholar-object.

        [label]:        <str> name
        [generator]:    <Generator> for generating samples (images + labels) from previous tasks
        [solver]:       <Solver> for classifying images'''
        super().__init__()
        self.label = label
        self.generator = generator
        self.solver = solver

    def train_with_replay(self, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32,
            generator_iterations=2000,
            generator_training_callbacks=None,
            solver_iterations=1000,
            solver_training_callbacks=None,
            collate_fn=None,
            valid_proportion=1./6):
        # [scholar] and [previous datasets] cannot both be given
        # (i.e., replay is either "generated" or "exact", or "none").
        both_selected = all([scholar is not None, bool(previous_datasets)])
        assert not both_selected, '[scholar] and [previous_datasets] cannot both be given'

        # train the generator of the scholar.
        self._train_batch_trainable_with_replay(
            self.generator, dataset, scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=generator_iterations,
            training_callbacks=generator_training_callbacks,
            collate_fn=collate_fn,
            valid_proportion=valid_proportion
        )

        # train the solver of the scholar.
        self._train_batch_trainable_with_replay(
            self.solver, dataset, scholar,
            previous_datasets=previous_datasets,
            importance_of_new_task=importance_of_new_task,
            batch_size=batch_size,
            iterations=solver_iterations,
            training_callbacks=solver_training_callbacks,
            collate_fn=collate_fn,
            valid_proportion=valid_proportion
        )

    @property
    def name(self):
        return self.label

    def sample(self, size):
        x = self.generator.sample(size)
        y = self.solver.solve(x)
        return x.data, y.data

    def _train_batch_trainable_with_replay(self, trainable, dataset, scholar=None, previous_datasets=None,
            importance_of_new_task=.5, batch_size=32, iterations=1000, training_callbacks=None, collate_fn=None,
            valid_proportion=0.):
        # do not train the model when given non-positive iterations.
        if iterations <= 0:
            return

        # create data loaders.
        data_loader = iter(utils.get_data_loader(
            dataset, batch_size, cuda=self._is_on_cuda(),
            collate_fn=collate_fn, valid_proportion=valid_proportion
        ))
        data_loader_previous = iter(utils.get_data_loader(
            ConcatDataset(previous_datasets), batch_size,
            cuda=self._is_on_cuda(), collate_fn=collate_fn, valid_proportion=valid_proportion
        )) if previous_datasets else None

        # define a tqdm progress bar.
        progress = tqdm(range(1, iterations+1))

        for batch_index in progress:
            cuda = self._is_on_cuda()

            # decide from where to sample the training data.
            from_scholar = scholar is not None
            from_previous_datasets = bool(previous_datasets)

            # sample the real training data.
            x, y = next(data_loader)
            x = Variable(x).cuda() if cuda else Variable(x)
            y = Variable(y).cuda() if cuda else Variable(y)

            # sample the replayed training data.
            if from_previous_datasets:
                x_, y_ = next(data_loader_previous)
            elif from_scholar:
                x_, y_ = scholar.sample(batch_size)
            else:
                x_ = y_ = None

            # wrap the replayed training data in (cuda-)Variables.
            if x_ is not None and y_ is not None:
                x_ = Variable(x_).cuda() if cuda else Variable(x_)
                y_ = Variable(y_).cuda() if cuda else Variable(y_)

            # train the model with a batch.
            result = trainable.train_a_batch(
                x, y, x_=x_, y_=y_,
                importance_of_new_task=importance_of_new_task
            )

            # fire the callbacks on each iteration.
            for callback in (training_callbacks or []):
                callback(trainable, progress, batch_index, result)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
