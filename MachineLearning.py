import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from copy import deepcopy
from random import sample
from numpy import Infinity, zeros
from math import floor
import logging
from Dataset import ImageDataset
from File import Arff
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from torchvision import models, transforms
from Keys import ConvNeuralNetwork
from Forms import TrainingForm, LayerWeightsExtractionForm

class ModelLearningSummary:
    def __init__(self, epochs: int) -> None:
        self.best_accuracy = 0.0
        self.best_loss = Infinity
        self.training = PhaseHistory(epochs)
        self.eval = PhaseHistory(epochs)


class PhaseHistory:
    def __init__(self, epochs: int) -> None:
        self.accuracy = zeros(epochs)
        self.loss = zeros(epochs)


class _CrossValidationLearningSummary:
    def __init__(self, epochs: int) -> None:
        self.best_accuracy = 0.0
        self.best_loss = Infinity
        self._epochs = epochs
        self.folds = []

    def get_average_performance(self) -> ModelLearningSummary:
        summary = ModelLearningSummary(self._epochs)
        fold_count = len(self.folds)

        for epoch in range(self._epochs):
            for fold in self.folds:
                summary.training.accuracy[epoch] += fold.training.accuracy[epoch]
                summary.training.loss[epoch] += fold.training.loss[epoch]
                summary.eval.accuracy[epoch] += fold.eval.accuracy[epoch]
                summary.eval.loss[epoch] += fold.eval.loss[epoch]
            summary.training.accuracy[epoch] /= fold_count
            summary.training.loss[epoch] /= fold_count
            summary.eval.accuracy[epoch] /= fold_count
            summary.eval.loss[epoch] /= fold_count
        return summary


class LogAgent():
    def __init__(self, filename) -> None:
        self._filename = filename
        self._logger = LogAgent._get_logger(filename)
        if self._logger:
            self._logger.setLevel(logging.DEBUG)
           
    def __str__(self) -> str:
        return str(self._logger) 
    
    def _get_logger(filename:str):
        if filename == None: return None
        logger = logging.getLogger(filename)
        # Create handlers
        f_handler = logging.FileHandler(filename + ".log")
        f_handler.setLevel(logging.DEBUG)
        f_format = logging.Formatter('%(message)s')
        f_handler.setFormatter(f_format)
        print(f_handler)
        logger.addHandler(f_handler)
        return logger

    def write(self, txt: str, print_text=True) -> None:
        if (print_text):
            print(txt)
        if (self._logger == None):
            return
        self._logger.debug(txt)


class DataHandler():
    @staticmethod
    def split_data(data: list, train_prct: float, eval_prct: float):
        """
        Returns data splitted in training, evaluation and test groups

        Parameters
        ---
        data : list
            List of data to be split
        train_prct : float
            Percentage of data to be included into training set (0 to 1)
        eval_prct : float
            Percentage of data to be included into evaluation set (0 to 1)

        Returns
        ---
        tuple[list, list, list]
            The training, evaluation and test groups respectively
        """
        sample_count = len(data)
        train_size = floor(sample_count * train_prct)
        eval_size = floor(sample_count * eval_prct)

        return data[:train_size], data[train_size:train_size + eval_size], data[train_size + eval_size:]

    @staticmethod
    def create_folds(data, fold_count: int):
        """
                Separate data into folds

                Parameters
                ---
                data : list | ImageDataset
                    Data to be split into folds
                fold_count : int
                    Number of folds to split the data

                Returns
                ---
                list
                    List with each fold (other lists) of data
            """

        ref_list = data.images if type(data) == ImageDataset else data

        # shuffle the list
        shuffled_data = sample(ref_list, len(data))
        # get fold size
        fold_size = len(shuffled_data) // fold_count
        # separate in equally sized folds
        folds = [shuffled_data[i*fold_size:(i+1)*fold_size]
                 for i in range(fold_count)]
        # count left out data
        left_out = len(shuffled_data) % fold_count
        # distribute left out data into folds
        for idx in range(left_out):
            folds[idx].append(shuffled_data[-1*(idx+1)])

        return folds

    @staticmethod
    def get_fold_groups(current_fold_index: int, all_folds: list):
        """
            Aggregate fold list into separate groups containing training data and evaluation data
            according to which is the current fold. 

            Parameters
            ---
            current_fold_index : int
                Index for the current fold (evaluation data)
            all_folds : list[list]
                List with all the separated folds

            Returns
            ---
            tuple[list,list]
                Respectively the evaluation and training groups of data
        """
        train = []
        for idx in range(len(all_folds)):
            if idx != current_fold_index:
                train += all_folds[idx]
        return all_folds[current_fold_index], train

    @staticmethod
    def create_batches(data: list, max_batch_size: int):
        """
            Separete data into chunks of a given size

            Parameters
            ---
            data : list
                List of data to be split into batches
            max_batch_size : int
                Maximum expected size of the output batches, 0 means there's no limit so there's only 1 batch with all the data

            Returns
            ---
            list[list]
                List with the splitted data into separated lists
        """
        if max_batch_size <= 0:
            return [data]
        batch_count = floor(len(data)/max_batch_size) + 1
        batches = []
        for batch_idx in range(batch_count):
            start_idx = batch_idx*max_batch_size
            end_idx = start_idx + max_batch_size
            batches.append(data[start_idx:end_idx])
        return [batch for batch in batches if len(batch) > 0]


class Trainer():
    def _copy_weights(model: nn.Module):
        """
            Get a copy of the weights from a neural network model

            Parameters
            ---
            model : torch.nn.Module
                Neural network from which the weights should be extracted

            Returns
            ---
            OrderedDict[str,Tensor]
                Weights for a neural network                
        """
        return deepcopy(model.state_dict())

    def _evaluate_model(model: nn.Module, data: list, labels: list, error_fn, device: str):
        """
            Evaluate neural network classification model based on the supplied data and error function

            Parameters
            ---
            model : torch.nn.Module
                Neural network model to evaluate
            data : list
                List with evaluation data
            labels : list
                List of labels expected for the evaluation data (with matching order)
            error_fn  : torch.nn error function
                Error function to be applied on the evaluation
            device : str
                Device used to perform the computing (either cpu or cuda)

            Returns
            ---
                tuple[float, float, Tensor]
                Respectively the value for the evaluation loss, acuracy and 
                information for the error function to be used on the backpropagation
        """
        # get correct labels
        _, correct_labels = torch.max(labels.to(device), 1)

        # predict data
        output = model(data.to(device))
        # get error
        loss_output = error_fn(output, labels.to(device))

        # define predictions as a single value for the highest probability output
        _, predictions = torch.max(output, 1)

        # count the correct predictions
        correct = torch.sum(predictions == correct_labels)
        # calculate loss
        loss = loss_output.item() / len(data)
        # calculate model accuracy
        accuracy = float(correct) / len(data)

        return loss, accuracy, loss_output

    def _process_input(model: nn.Module,
                       data: list,
                       labels: list,
                       error_fn,
                       optimization_fn: optim.Optimizer = None,
                       device=torch.device("cpu"),
                       max_batch_size: int = 0,
                       learn=False,):
        """
            Process input data in batches using a neural network model

            Parameters
            ---
            model : torch.nn.Module
                Neural network model to evaluate
            data : list
                List with input data
            labels : list
                List of labels expected for the data (with matching order)
            error_fn  : torch.nn error function
                Error function to be applied on the processing
            optimization_fn : torch.optim.Optimizer, optional
                Optimization function used on training to realize the backpropagation.
                Only required if training while processing
            device : torch.device
                Device used to perform the computing (default is cpu).
            max_batch_size : int, optional
                Maximum size of the batches processed (default is 0 meaning there's no limit)
            learn : bool, optional
                Whether or not to train the model while processing (default is False)

            Returns
            ---
                tuple[float, float, Tensor]
                Respectively the value for the evaluation loss, acuracy and 
                information for the error function to be used on the backpropagation
        """

        # define processing mode on the neural network
        if learn:
            model.train()
        else:
            model.eval()

        # split input into batches to avoid memory overload
        data_input = DataHandler.create_batches(data, max_batch_size)
        data_labels = DataHandler.create_batches(labels, max_batch_size)
        batch_count = len(data_input)
        loss = 0
        accuracy = 0

        # process batches
        for batch_idx in range(batch_count):

            # create stacked input from data
            batch_input = torch.stack([img.get_tensor()
                                       for img in data_input[batch_idx]])
            batch_labels = torch.stack(data_labels[batch_idx])

            if learn:
                # if learning, clear the gradient to get a brand new adjustment
                # with clean information
                optimization_fn.zero_grad()

            # evaluate model with input data
            batch_loss, batch_accuracy, loss_info = Trainer._evaluate_model(model,
                                                                            batch_input,
                                                                            batch_labels,
                                                                            error_fn=error_fn,
                                                                            device=device)

            # hold loss and accuracy values to summarize later
            accuracy += batch_accuracy
            loss += batch_loss

            if learn:
                # if learning, apply the adjustment with backpropagation
                loss_info.backward()
                optimization_fn.step()

            if device == torch.device("cuda:0"):
                # if using gpu, clear the cache of used data to save up on
                # memory use
                torch.cuda.empty_cache()

        return model, accuracy/batch_count, loss/batch_count

    @staticmethod
    def train(model: nn.Module,
              dataset: ImageDataset,
              train: list,
              eval: list,
              epochs: int,
              optimizer=optim.SGD,
              weight_decay: float = 0,
              learning_rate: float = 0.1,
              error_fn=nn.MSELoss(),
              plot_acc=True,
              plot_loss=True,
              use_gpu=False,
              max_batch_size=0,
              learning_rate_drop=0,
              learning_rate_drop_step_size=0,
              is_notebook_env=False,
              log_filename: str = None,
              log: LogAgent = None):
        """
        Trains a neural network

        Parameters
        ---
        model : torch.nn.Module
            Neural network model to train
        dataset: ImageDataset
            Dataset to be used as a reference for the images
        train : list[ImageDatasetEntry]
            List of ImageDatasetEntry used for the training process
        eval : list[ImageDatasetEntry]
            List of ImageDatasetEntry used for the evaluation process
        epochs : int
            Number of iterations for the training
        optimizer : torch.optim.Optimizer, optional
            Function used to calculate the backpropagation on training phase (default
            is SGD, also known as stochastic gradient descent)
        weight_decay : float, option
            Decay multiplier to be applied to weight adjustments on the optimization. 
            If 0, no decay will be considered (default is 0)
        learning_rate : float, optional
            Learning rate used to calculate the adjustments on backpropagation (default
            is 0.1)
        error_fn : torch.nn error function, optional
            Function used to obtain the error parameters on the processing, used to
            acquire the data necessary for the adjustments (default is MSE, also known
            as mean square error).
        plot_acc : bool, optional
            Whether or not to plot the accuracy chart after the training (default is True)
        plot_loss : bool, optional
            Whether or not to plot the loss chart after the training (default is True)
        use_gpu : bool, optional
            Whether or not to use the GPU, if available, on training (default is False)
        max_batch_size : int, optional
                Maximum size of the batches processed (default is 0 meaning there is no limit)
        learning_rate_drop: float, optional
            Multiplier to reduce learning rate afte a given number of epochs. If 0 there
            will not be a drop on the learning rate (default is 0).
        learning_rate_drop_step_size : int, optional 
            Number of steps between drops on the learning rate. If 0 there will not be a
                drop on the learning rate (default is 0).
        is_notebook_env : bool, optional
            Wheter to use python notebook specific functions
        Returns
        ---
        tuple[torch.Module, ModelLearningSummary]
            Respectively the trained neural network, a history with it's accuracy, it's loss, it's 
            history of accuracy on training, it's history of loss on training, it's 
            history of accuracy on evaluation, it's history of loss on evaluation

        """

        # setup environment usage
        can_use_gpu = use_gpu and torch.cuda.is_available()
        progress_bar = tqdm_notebook if is_notebook_env else tqdm
        device = torch.device("cuda:0" if can_use_gpu else "cpu")
        training_model = model.to(device)

        training_log = log
        if (log == None):
            training_log = LogAgent(log_filename)

        # start best weights as empty
        best_weights = Trainer._copy_weights(training_model)

        # Start epoch history track
        history = ModelLearningSummary(epochs)

        # get folds for training and evaluating
        eval_labels = [dataset.get_expected_tensor(img) for img in eval]
        train_labels = [dataset.get_expected_tensor(img) for img in train]

        # define optimization function
        optimization_fn = optimizer(training_model.parameters(
        ), lr=learning_rate, weight_decay=weight_decay)

        # define decay on learning rate if any is required
        learning_rate_drop_fn = None
        if (learning_rate_drop_step_size > 0 and learning_rate_drop != 0):
            learning_rate_drop_fn = optim.lr_scheduler.StepLR(
                optimization_fn, step_size=learning_rate_drop_step_size, gamma=learning_rate_drop)

        progress = progress_bar(range(epochs))
        for epoch in progress:
            training_log.write(f"\n  Epoch {epoch+1}", False)

            # Training phase
            progress.set_description("Learning")

            training_model, accuracy, loss = Trainer._process_input(training_model,
                                                                    train,
                                                                    train_labels,
                                                                    error_fn,
                                                                    optimization_fn=optimization_fn,
                                                                    device=device,
                                                                    max_batch_size=max_batch_size,
                                                                    learn=True)

            training_log.write(f'    Training - Loss: {loss:.4f} Acc: {accuracy:.4f}',
                               False)
            # Save data to training history
            history.training.accuracy[epoch] = accuracy
            history.training.loss[epoch] = loss

            # Evaluation phase
            progress.set_description("Evaluating")
            _, accuracy, loss = Trainer._process_input(training_model,
                                                       eval,
                                                       eval_labels,
                                                       error_fn,
                                                       optimization_fn=optimization_fn,
                                                       device=device,
                                                       max_batch_size=max_batch_size,
                                                       learn=False)

            if accuracy > history.best_accuracy:
                history.best_accuracy = accuracy
                history.best_loss = loss
                best_weights = Trainer._copy_weights(training_model)

            training_log.write(
                f'    Evaluation - Loss: {loss:.4f} Acc: {accuracy:.4f}', False)
            # Save data to eval history
            history.eval.accuracy[epoch] = accuracy
            history.eval.loss[epoch] = loss

            if (learning_rate_drop_fn != None):
                learning_rate_drop_fn.step()

        training_log.write(f'\n  [Training Complete] Best Loss: {history.best_loss:.4f} Best Acc: {history.best_accuracy:.4f}',
                           False)

        # Plot graphs
        if plot_acc:
            _, ax = plt.subplots()
            ax.plot(history.training.accuracy, 'r',
                    label="Average training acurracy")
            ax.plot(history.eval.accuracy, 'b',
                    label="Average evaluation accuracy")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Accuracy")
            ax.legend()
            plt.show()

        if plot_loss:
            _, ax = plt.subplots()
            ax.plot(history.training.loss, 'r', label="Average training loss")
            ax.plot(history.training.loss, 'b',
                    label="Average evaluation loss")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Loss")
            ax.legend()
            plt.show()

        # load best model weights
        training_model.load_state_dict(best_weights)

        return training_model, history

    @staticmethod
    def k_fold_training(model: nn.Module,
                        dataset: ImageDataset,
                        epochs: int,
                        k: int = 3,
                        optimizer=optim.SGD,
                        weight_decay: float = 0,
                        learning_rate: float = 0.1,
                        error_fn=nn.MSELoss(),
                        plot_acc=True,
                        plot_loss=True,
                        max_batch_size=0,
                        use_gpu=False,
                        learning_rate_drop=0,
                        learning_rate_drop_step_size=0,
                        log_filename: str = None,
                        is_notebook_env: bool = False):
        """
        Trains a neural network using k-fold cross validation

        Parameters
        ---
        model : torch.nn.Module
            Neural network model to train
        dataset : Dataset.ImageDataset
            Dataset of images to be used on the training
        epochs : int
            Number of iterations for the training
        k : int, optional
            Number of folds used on cross validation (default is 3)
        optimizer : torch.optim.Optimizer, optional
            Function used to calculate the backpropagation on training phase (default
            is SGD, also known as stochastic gradient descent)
        weight_decay : float, option
            Decay multiplier to be applied to weight adjustments on the optimization. 
            If 0, no decay will be considered (default is 0)
        learning_rate : float, optional
            Learning rate used to calculate the adjustments on backpropagation (default
            is 0.1)
        error_fn : torch.nn error function, optional
            Function used to obtain the error parameters on the processing, used to
            acquire the data necessary for the adjustments (default is MSE, also known
            as mean square error).
        plot_acc : bool, optional
            Whether or not to plot the accuracy chart after the training (default is True)
        plot_loss : bool, optional
            Whether or not to plot the loss chart after the training (default is True)
        use_gpu : bool, optional
            Whether or not to use the GPU, if available, on training (default is False)
        max_batch_size : int, optional
                Maximum size of the batches processed (default is 0 meaning there is no limit)
        learning_rate_drop: float, optional
            Multiplier to reduce learning rate afte a given number of epochs. If 0 there
            will not be a drop on the learning rate (default is 0).
        learning_rate_drop_step_size : int, optional 
            Number of steps between drops on the learning rate. If 0 there will not be a
                drop on the learning rate (default is 0).
        is_notebook_env : bool, optional
            Wheter to use python notebook specific functions

        Returns
        ---
        torch.Module, CrossValidationLearningSummary
            The best trained neural network from all the folds and a summary for its training
        """

        log = LogAgent(log_filename)

        # save inital state to recover between folds
        untrained = Trainer._copy_weights(model)

        # setup initial best
        best_weights = Trainer._copy_weights(model)

        # Start epoch history track
        history = _CrossValidationLearningSummary(epochs)

        # create folds from data
        folds = DataHandler.create_folds(dataset, k)

        log.write(f"Got {len(folds)} folds with {len(folds[0])} samples")

        training_start_timestamp = time.time()

        # train each fold
        for current_fold_idx in range(len(folds)):
            log.write(f"\nProcessing Fold {current_fold_idx + 1}")
            # get training and evaluating data from fold list
            eval_input, training_input = DataHandler.get_fold_groups(
                current_fold_idx, folds)
            fold_start_timestamp = time.time()
            # train fold
            fold_best_model, fold_history = Trainer.train(
                model=model,
                dataset=dataset,
                train=training_input,
                eval=eval_input,
                epochs=epochs,
                optimizer=optimizer,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                error_fn=error_fn,
                max_batch_size=max_batch_size,
                use_gpu=use_gpu,
                plot_loss=False,
                plot_acc=False,
                learning_rate_drop=learning_rate_drop,
                learning_rate_drop_step_size=learning_rate_drop_step_size,
                log=log,
                is_notebook_env=is_notebook_env
            )
            fold_duration = time.time() - fold_start_timestamp

            log.write(f'\n  Fold {current_fold_idx+1} Training complete in {(fold_duration // 60):.0f}m {(fold_duration % 60):.0f}s',
                      False)
            # if is better than the previous, store it
            if fold_history.best_accuracy > history.best_accuracy:
                history.best_accuracy = fold_history.best_accuracy
                history.best_loss = fold_history.best_loss
                best_weights = Trainer._copy_weights(fold_best_model)

            # keep track of the performance
            history.folds.append(fold_history)

            # reset model weights
            model.load_state_dict(untrained)

        training_duration = time.time() - training_start_timestamp

        log.write(
            f'\nTraining complete in {(training_duration // 60):.0f}m {(training_duration % 60):.0f}s', False)

        # summarize the performance
        summary = history.get_average_performance()

        # Plot graphs
        if plot_acc:
            _, ax = plt.subplots()
            ax.plot(summary.training.accuracy, 'r',
                    label="Average training acurracy")
            ax.plot(summary.eval.accuracy, 'b',
                    label="Average evaluation accuracy")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Accuracy")
            ax.legend()
            plt.show()

        if plot_loss:
            _, ax = plt.subplots()
            ax.plot(summary.training.loss, 'r', label="Average training loss")
            ax.plot(summary.eval.loss, 'b', label="Average evaluation loss")
            ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Loss")
            ax.legend()
            plt.show()

        # load best model weights
        model.load_state_dict(best_weights)

        return model, history
            
class Helper():
    _LOAD_MODEL_FUNCTIONS = {
        ConvNeuralNetwork.RESNET_50: models.resnet50,
        ConvNeuralNetwork.DENSENET_121: models.densenet121,
        ConvNeuralNetwork.EFFICIENTNET_B2: models.efficientnet_b2
    }

    _PRETRAINED_MODEL_WEIGHTS = {
        ConvNeuralNetwork.RESNET_50: models.ResNet50_Weights.IMAGENET1K_V1,
        ConvNeuralNetwork.DENSENET_121: models.DenseNet121_Weights.IMAGENET1K_V1,
        ConvNeuralNetwork.EFFICIENTNET_B2: models.EfficientNet_B2_Weights.IMAGENET1K_V1
    }

    @classmethod
    def _load_form_model(self, model: ConvNeuralNetwork, is_transfer:bool):
        fn = self._LOAD_MODEL_FUNCTIONS[model]
        if is_transfer:
            return fn(weights=self._PRETRAINED_MODEL_WEIGHTS[model])
        return fn()

    @staticmethod
    def _update_model_output_size(model:nn.Module, model_id: ConvNeuralNetwork, output_size: int):
        if model_id == ConvNeuralNetwork.RESNET_50:
            num_feats = model.fc.in_features
            model.fc = nn.Linear(num_feats, output_size)

        elif model_id ==  ConvNeuralNetwork.DENSENET_121:
            num_feats = model.classifier.in_features
            model.classifier = nn.Linear(num_feats, output_size)

        elif model_id ==  ConvNeuralNetwork.EFFICIENTNET_B2:
            num_feats = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_feats, output_size)

    @staticmethod
    def _get_model_last_layer_size(model:nn.Module, model_id: ConvNeuralNetwork) -> int:
        if model_id == ConvNeuralNetwork.RESNET_50 or model_id == ConvNeuralNetwork.INCEPTION_V3:
            return model.fc.in_features  # 2048

        elif model_id ==  ConvNeuralNetwork.DENSENET_121:
            return model.classifier.in_features # 1024

        elif model_id ==  ConvNeuralNetwork.EFFICIENTNET_B2:
            return model.classifier[1].in_features # 1408
        
        elif model_id == ConvNeuralNetwork.VGG19:
            return model.classifier[6].in_features # 4096
       
    @staticmethod
    def _freeze_parameters(model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def _get_transform_functions(add_normalize:bool = False):
        functions = [transforms.ToTensor()]
        if add_normalize:
            functions.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        return transforms.Compose(functions)

    @staticmethod
    def _register_output_layer_hook(model:nn.Module, model_id: ConvNeuralNetwork, holder_dict: dict):
        # Define function to send input/output to holder with a specific name
        def get_input(name):
            def hook(_model, input, _output):
                aux_array = input[0].cpu().detach().numpy()
                aux_array = aux_array.flatten()
                holder_dict[name] = aux_array.tolist()

            return hook

        def get_output(name):
            def hook(_model, _input, output):
                aux_array = output.cpu().detach().numpy()
                aux_array = aux_array.flatten()
                holder_dict[name] = aux_array.tolist()

            return hook

        if (model_id == ConvNeuralNetwork.RESNET_50 
            or model_id == ConvNeuralNetwork.INCEPTION_V3
            or model_id == ConvNeuralNetwork.EFFICIENTNET_B2):
            model.avgpool.register_forward_hook(get_output(model_id))


        elif model_id ==  ConvNeuralNetwork.DENSENET_121:
            model.classifier.register_forward_hook(get_input(model_id))

        
        elif model_id == ConvNeuralNetwork.VGG19:
            model.classifier[5].register_forward_hook(get_output(model_id))

    @staticmethod
    def train_cnn(form: TrainingForm):
        class_list = ImageDataset.get_csv_available_classes(form.dataset.csv_path)

        # 1. Load model
        model = Helper._load_form_model(form.model_id, form.is_transfer)

        # 2. Freeze the training for all layers
        # Obs. This step only applies for transfer learning
        if form.is_freeze: Helper._freeze_parameters(model)


        # 3. Update output to match number of classes
        Helper._update_model_output_size(model, form.model_id, len(class_list))

        # 4. Create transforms for the data
        # Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
        # ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
        # using this values, but hey can be changed to values that best suits your case.
        transform_functions = Helper._get_transform_functions(add_normalize=form.is_transfer)

        # 5. Create dataset. This type can be found in the file Dataset.py of this package
        # and gets the path to a csv with the list of the images file names and the base path to the folder of the
        # images. If you don't have the csv already, you can use the 'createFolderContentCsv' function
        # from the file FileHandling.py.
        dataset = ImageDataset(
            form.dataset.csv_path,
            form.dataset.folder_path,
            class_list,
            transform=transform_functions,
        )

        # 6. Call the training function
        print("\nTraining...")
        trained_model, learning_history = Trainer.k_fold_training(
            model,
            dataset,
            k = form.training_parameters.number_of_folds,
            epochs = form.training_parameters.training_epochs,
            learning_rate = form.training_parameters.learning_rate,
            learning_rate_drop = form.training_parameters.learning_rate_drop, 
            learning_rate_drop_step_size= form.training_parameters.learning_rate_drop_frequency, 
            max_batch_size=5,
            plot_acc=False,
            plot_loss=False,
            log_filename=form.log_path,
            use_gpu=torch.cuda.is_available()
        )

        # 7. Save trained model (Optional)
        torch.save(trained_model.state_dict(), form.output_path)

    @staticmethod
    def extract_output_layer_values(form:LayerWeightsExtractionForm):
        # 1. Create dictionary to hold outputs
        data_holder = {}

        # 2. Load database
        # Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
        # ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
        # using this values, but hey can be changed to values that best suits your case.
        transform_functions = Helper._get_transform_functions(add_normalize=False)
        class_list = ImageDataset.get_csv_available_classes(form.dataset.csv_path)
        dataset = ImageDataset(form.dataset.csv_path,
                            form.dataset.folder_path,
                            class_list,
                            transform=transform_functions)

        img_count = len(dataset)  # get image count

        # 3. Load CNN
        model = Helper._load_form_model(form.model_id, form.is_transfer)

        if form.model_path:
            model.load_state_dict(torch.load(form.model_path))
        model.eval()  # set it to evaluation mode

        # 3.1. If GPU is available, send cnn to it
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model.cuda()

        # 3.2 Get layer output size 
        output_size = Helper._get_model_last_layer_size(model, form.model_id)

        # 4. Register hook to desired layer to hold it's output
        Helper._register_output_layer_hook(model, form.model_id, data_holder)

        # 5. Process each image and get the desired layers output
        # create placeholder output list
        output = np.zeros((img_count, output_size + 1)).tolist()
        progress = tqdm(range(img_count))
        for idx in progress:
            progress.set_description(dataset[idx].filename)

            # create fake batch with single input
            img = dataset[idx].get_tensor().unsqueeze(0)
            # if cuda is available send input to it
            if use_gpu:
                img = img.cuda()

            # give input to cnn
            out = model(img)

            if use_gpu:
                torch.cuda.empty_cache()

            # get all the data collected in the holder
            img_output = data_holder[form.model_id]
            # append image class to the end of the output
            img_output.append(dataset[idx].class_id)

            # store output
            output[idx] = img_output

        # 6. Create the names for the extracted variables
        names = [f"{form.model_id.value}_output_value_{i}" for i in range(1, output_size + 1)]

        # 7. Write arff file
        data = Arff(relation=F"{form.model_id.value}-output",
                    entries=output,
                    classes=class_list,
                    attrs=names, 
                    attr_types=['numeric' for _ in names]
                )
        
        path, filename = os.path.split(form.output_path)
        data.save(filename, path)
