#!/usr/bin/env python
# encoding: utf-8

import copy
import os
import time

import numpy as np
import torch


class GenericTrainer(object):
    """
    Class to train a generic NN; all the parameters are provided in configs

    Attributes
    ----------
    network: :py:class:`torch.nn.Module`
        The network to train
    batch_size: int
        The size of your minibatch
    optimizer: :py:class:`torch.optim.Optimizer`
        Optimizer object to be used. Initialized in the config file.

    device: str
        Device which will be used for training the model
    """

    def __init__(self, network, optimizer, compute_loss, learning_rate=0.0001, batch_size=64, device='cpu',
                 do_crossvalidation=False, save_interval=5):
        """ Init function . The layers to be adapted in the network is selected and the gradients are set to `True`
        for the  layers which needs to be adapted.

        Parameters
        ----------
        network: :py:class:`torch.nn.Module`
            The network to train
        batch_size: int
            The size of your minibatch
        device: str
            Device which will be used for training the model
        do_crossvalidation: bool
            If set to `True`, performs validation in each epoch and stores the best model based on validation loss.
        """
        self.network = network
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.compute_loss = compute_loss
        self.device = device
        self.learning_rate = learning_rate
        self.save_interval = save_interval

        self.do_crossvalidation = do_crossvalidation

        if self.do_crossvalidation:
            phases = ['train', 'val']
        else:
            phases = ['train']
        self.phases = phases

        # Move the network to device
        self.network.to(self.device)

    # Setting the gradients to true for the layers which needs to be adapted

    def load_model(self, model_filename):
        """Loads an existing model

        Parameters
        ----------
        model_filename: str
            The filename of the model to load

        Returns
        -------
        start_epoch: int
            The epoch to start with
        start_iteration: int
            The iteration to start with
        losses: list(float)
            The list of losses from previous training

        """

        cp = torch.load(model_filename)
        self.network.load_state_dict(cp['state_dict'])
        start_epoch = cp['epoch']
        start_iter = cp['iteration']
        losses = cp['loss']
        return start_epoch, start_iter, losses

    def save_model(self, output_dir, epoch=0, iteration=0, losses=None):
        """Save the trained network

        Parameters
        ----------
        output_dir: str
            The directory to write the models to
        epoch: int
            the current epoch
        iteration: int
            the current (last) iteration
        losses: list(float)
                The list of losses since the beginning of training

        """

        saved_filename = f'model_{epoch}_{iteration}.pth'
        saved_path = os.path.join(output_dir, saved_filename)
        print(f'Saving model to {saved_path}')
        cp = {'epoch': epoch,
              'iteration': iteration,
              'loss': losses,
              'state_dict': self.network.cpu().state_dict()
              }
        torch.save(cp, saved_path)

        self.network.to(self.device)

    def train(self, dataloader, n_epochs=25, output_dir='out', model=None):
        """Performs the training.

        Parameters
        ----------
        dataloader: :py:class:`torch.utils.data.DataLoader`
            The dataloader for your data
        n_epochs: int
            The number of epochs you would like to train for
        output_dir: str
            The directory where you would like to save models
        model: str
            The path to a pretrained model file to start training from; this is the PAD model; not the LightCNN model

        """

        # if model exists, load it
        if model is not None:
            start_epoch, start_iter, losses = self.load_model(model)
            print(
                f'Starting training at epoch {start_epoch}, iteration {start_iter} - last loss value is {losses[-1]}')
        else:
            start_epoch = 0
            start_iter = 0
            losses = []
            print('Starting training from scratch')

        for name, param in self.network.named_parameters():

            if param.requires_grad:
                print(f'Layer to be adapted from grad check : {name}')

        # setup optimizer

        self.network.train(True)

        best_model_wts = copy.deepcopy(self.network.state_dict())

        best_loss = float("inf")

        # let's go
        for epoch in range(start_epoch, n_epochs):

            # in the epoch

            train_loss_history = []

            val_loss_history = []

            for phase in self.phases:

                if phase == 'train':
                    self.network.train()  # Set model to training mode
                else:
                    self.network.eval()  # Set model to evaluate mode

                for i, data in enumerate(dataloader[phase]):

                    if i >= start_iter:

                        start = time.time()

                        # get data from dataset

                        img, labels = data

                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):

                            loss = self.compute_loss(self.network, img, labels, self.device)

                            if phase == 'train':

                                loss.backward()

                                self.optimizer.step()

                                train_loss_history.append(loss.item())
                            else:

                                val_loss_history.append(loss.item())

                        end = time.time()

                        print(
                            f"[{epoch}/{n_epochs}][{i}/{len(dataloader[phase])}] => Loss = {loss.item()} (time spent: {(end - start)}), Phase {phase}")

                        losses.append(loss.item())

                epoch_train_loss = np.mean(train_loss_history)

                print(f"Train Loss : {epoch_train_loss}  epoch : {epoch}")

                if self.do_crossvalidation:

                    epoch_val_loss = np.mean(val_loss_history)

                    print(f"Val Loss : {epoch_val_loss}  epoch : {epoch}")

                    if phase == 'val' and epoch_val_loss < best_loss:
                        print(
                            f"New val loss : {epoch_val_loss} is better than old: {best_loss}, copying over the new weights")

                        best_loss = epoch_val_loss

                        best_model_wts = copy.deepcopy(self.network.state_dict())

            # Logging
            if self.do_crossvalidation:

                info = {'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
            else:

                info = {'train_loss': epoch_train_loss}

                # scalar logs

            for tag, value in info.items():
                print(tag, value, epoch + 1)

            # Log values and gradients of the parameters (histogram summary)

            # for tag, value in self.network.named_parameters():
            #     tag = tag.replace('.', '/')
            #     try:
            #         print(tag, value.data.cpu().numpy(), epoch + 1)
            #         print(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
            #     except Exception:
            #         pass

            # /Logging

            # do stuff - like saving models
            print(f"EPOCH {epoch + 1} DONE")

            # comment it out after debugging

            if (epoch + 1) == n_epochs or epoch % self.save_interval == 0:  # save the last model, and the ones
                # in the specified interval
                print(f"saving to {output_dir}")
                self.save_model(output_dir, epoch=(epoch + 1), iteration=0, losses=losses)

        # load the best weights

        self.network.load_state_dict(best_model_wts)

        # best epoch is 0

        self.save_model(output_dir, epoch=0, iteration=0, losses=losses)
