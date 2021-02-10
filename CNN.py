import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import copy
import random


def trainCrossValidation(model, dataset, k:int, error_criterion, optmization_algorithm, epochs:int, plot_acc = False, plot_loss = False):
    """ Trains a torchvision model using k-folds cross-validation.

    Args:
        model(torchvision.models): Neural network model.
        dataset(Dataset.ImageDataset): Dataset to train.
        k(int): Number of folds, must be at least 2.
        error_criterion(torch.nn.modules.loss): Error function for the training.
        optmization_algorithm(torch.optim): Optimization rule for backpropagation.
        epochs(int): Number of training iteration for each fold.
        plot_acc(bool): Plot average epoch accuracy for training and evaluation after training is complete.
        plot_loss(bool): Plot average epoch loss for training and evaluation after training is complete.

    Output:
        Trained model with the weights that got the highest accuracy in the evaluation while training
    """
    # Gets execution start timestamp
    since = time.time()
    
    #Start epoch history track
    train_acc_hist = []
    train_loss_hist = []
    eval_acc_hist = []
    eval_loss_hist = []

    #Transfer model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Save untrained model to reset every fold
    untrained_model_weights = copy.deepcopy(model.state_dict())

    # Starts best model and accuracy to current values
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    img_count = len(dataset) # get image count
    img_idx_list = [i for i in range(img_count)] # create list with image indexes
    random.shuffle(img_idx_list) # shuffle the list

    # Separate data in folds
    fold_size = len(img_idx_list) // k # get fold size
    folds = [img_idx_list[i*fold_size:(i+1)*fold_size] for i in range(k)] # separate in equally sized folds
    left_out = len(img_idx_list) % k # count left out indexes
    # distribute left out indexes into folds
    for idx in range(left_out):
        folds[idx].append(img_idx_list[-1*(idx+1)])

    for current_fold in range(k):
        fold_start_time = time.time()
        fold_best_acc = 0.0
        fold_train_acc_hist = []
        fold_train_loss_hist = []
        fold_eval_acc_hist = []
        fold_eval_loss_hist = []

        # get list with all other folds
        other_folds = []
        for i in range(k):
            if i == current_fold:
                continue
            other_folds += folds[i]
        # Create tensor batch with training images and their labels
        training_inputs = []
        training_labels = []
        for i in range(len(other_folds)):
            training_inputs.append(dataset[other_folds[i]])
            training_labels.append(dataset.getExpected(other_folds[i]))
        training_inputs = torch.stack(training_inputs)
        training_labels = torch.stack(training_labels)
        
        # Create tensor batch with evaluation images and their labels
        eval_inputs = []
        eval_labels = []
        for i in range(len(folds[current_fold])):
            eval_inputs.append(dataset[folds[current_fold][i]])
            eval_labels.append(dataset.getExpected(folds[current_fold][i]))    
        eval_inputs = torch.stack(eval_inputs)
        eval_labels = torch.stack(eval_labels)

        # Start epoch iterations
        for epoch in range(epochs):
            print('-' * 10)
            print('Fold {}/{} - Epoch {}/{}'.format(current_fold +1, k, epoch + 1, epochs))
            epoch_start_time = time.time()

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    # Get inputs for phase
                    input_data = training_inputs
                    input_labels = training_labels
                else:
                    model.eval()   # Set model to evaluate mode
                    # Get inputs for phase
                    input_data = eval_inputs
                    input_labels = eval_labels

                # Transfer inputs to GPU if available
                if torch.cuda.is_available():
                    input_data, input_labels = input_data.cuda(), input_labels.cuda()

                # zero the parameter gradients
                optmization_algorithm.zero_grad()

                # Process the input
                output = model(input_data)
                # Get it's predictions                
                _, preds = torch.max(output, 1)
                # Calculate batch loss
                loss = error_criterion(output, input_labels)

                # If training apply backward + optimize to adust weights
                if phase == 'train':
                    loss.backward()
                    optmization_algorithm.step()

                # Get correct predictions
                _,correct_labels = torch.max(input_labels,1)

                # Get epoch global loss, count correct predictions and get accuracy
                epoch_correct_evals = torch.sum(preds == correct_labels)
                epoch_loss = loss.item() / len(input_data)
                epoch_acc = float(epoch_correct_evals) / len(input_data)

                # Save data to history
                if phase == 'val':
                    fold_eval_acc_hist.append(epoch_acc)
                    fold_eval_loss_hist.append(epoch_loss)
                else:
                    fold_train_acc_hist.append(epoch_acc)
                    fold_train_loss_hist.append(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # If accuray of evaluation set increased, save the model as new best
                if phase == 'val':
                    if epoch_acc > fold_best_acc:
                        fold_best_acc = epoch_acc
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_weights = copy.deepcopy(model.state_dict())

            epoch_time_elapsed = time.time() - epoch_start_time
            print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))    
            print()

        # reset model weights
        model.load_state_dict(untrained_model_weights)
        
        time_elapsed = time.time() - fold_start_time
        print('Fold {} Training complete in {:.0f}m {:.0f}s'.format(current_fold+1, time_elapsed // 60, time_elapsed % 60))
        print('Fold {} Best val Acc: {:4f}'.format(current_fold+1, fold_best_acc))

        train_acc_hist.append(fold_train_acc_hist)
        train_loss_hist.append(fold_train_loss_hist)
        eval_acc_hist.append(fold_eval_acc_hist)
        eval_loss_hist.append(fold_eval_loss_hist)

    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # Get average curves
    avg_train_acc = []
    avg_train_loss = []
    avg_eval_acc = []
    avg_eval_loss= []
    for epoch in range(epochs):
        train_acc = 0.0
        train_loss = 0.0
        eval_acc = 0.0
        eval_loss = 0.0
        for fold in range(k):
            train_acc += train_acc_hist[fold][epoch]
            train_loss += train_loss_hist[fold][epoch]
            eval_acc += eval_acc_hist[fold][epoch]
            eval_loss += eval_loss_hist[fold][epoch]
        avg_train_acc.append(train_acc/k)
        avg_train_loss.append(train_loss/k)
        avg_eval_acc.append(eval_acc/k)
        avg_eval_loss.append(eval_loss/k)


    # Plot graphs
    if plot_acc:
        fig, ax = plt.subplots()      
        ax.plot(avg_train_acc, 'r', label="Average training acurracy")
        ax.plot(avg_eval_acc, 'b', label="Average evaluation accuracy")
        ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Accuracy")
        ax.legend()
        plt.show()
    
    if plot_loss:
        fig, ax = plt.subplots()
        ax.plot(avg_train_loss, 'r', label="Average training loss")
        ax.plot(avg_eval_loss, 'b', label="Average evaluation loss")
        ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Loss")
        ax.legend()
        plt.show()

    # load best model weights
    model.load_state_dict(best_model_weights)

    return model

def train(model, dataset, train_percentage:float, error_criterion, optmization_algorithm, epochs:int, plot_acc = False, plot_loss = False):
    """ Trains a torchvision model using a percentage of the dataset to train and the rest to evaluate the training.

    Args:
        model(torchvision.models): Neural network model.
        dataset(Dataset.ImageDataset): Dataset to train.
        traing_percentage(float): 0 to 1 percentage of the data that should be used to train the model.
        error_criterion(torch.nn.modules.loss): Error function for the training.
        optmization_algorithm(torch.optim): Optimization rule for backpropagation.
        epochs(int): Number of training iteration for each fold.
        plot_acc(bool): Plot average epoch accuracy for training and evaluation after training is complete.
        plot_loss(bool): Plot average epoch loss for training and evaluation after training is complete.

    Output:
        Trained model with the weights that got the highest accuracy in the evaluation while training
    """

    # Gets execution start timestamp
    since = time.time()
    
    #Start epoch history track
    train_acc_hist = []
    train_loss_hist = []
    eval_acc_hist = []
    eval_loss_hist = []

    #Transfer model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Starts best model and accuracy to current values
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    img_count = len(dataset) # get image count
    img_idx_list = [i for i in range(img_count)] # create list with image indexes
    random.shuffle(img_idx_list) # shuffle the list

    training_separator_idx = int(img_count * train_percentage) # get separation index

    # Create tensor batch with training images and their labels
    training_inputs = []
    training_labels = []
    for i in range(training_separator_idx):
        training_inputs.append(dataset[img_idx_list[i]])
        training_labels.append(dataset.getExpected(img_idx_list[i]))
    training_inputs = torch.stack(training_inputs)
    training_labels = torch.stack(training_labels)
    
    # Create tensor batch with evaluation images and their labels
    eval_inputs = []
    eval_labels = []
    for i in range(img_count-training_separator_idx):
        eval_inputs.append(dataset[img_idx_list[training_separator_idx + i]])
        eval_labels.append(dataset.getExpected(img_idx_list[training_separator_idx + i]))    
    eval_inputs = torch.stack(eval_inputs)
    eval_labels = torch.stack(eval_labels)

    # Start epoch iterations
    for epoch in range(epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        epoch_start_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # Get inputs for phase
                input_data = training_inputs
                input_labels = training_labels
            else:
                model.eval()   # Set model to evaluate mode
                # Get inputs for phase
                input_data = eval_inputs
                input_labels = eval_labels

            # Transfer inputs to GPU if available
            if torch.cuda.is_available():
                input_data, input_labels = input_data.cuda(), input_labels.cuda()

            # zero the parameter gradients
            optmization_algorithm.zero_grad()

            # Process the input
            output = model(input_data)
            # Get it's predictions                
            _, predictions = torch.max(output, 1)
            # Calculate batch loss
            loss = error_criterion(output, input_labels)

            # If training apply backward + optimize to adust weights
            if phase == 'train':
                loss.backward()
                optmization_algorithm.step()

            # Get correct predictions
            _,correct_labels = torch.max(input_labels,1)

            # Get epoch global loss, count correct predictions and get accuracy
            epoch_correct_evals = torch.sum(predictions == correct_labels)
            epoch_loss = loss.item() / len(input_data)
            epoch_acc = float(epoch_correct_evals) / len(input_data)

            # Save data to history
            if phase == 'val':
              eval_acc_hist.append(epoch_acc)
              eval_loss_hist.append(epoch_loss)
            else:
              train_acc_hist.append(epoch_acc)
              train_loss_hist.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # If accuray of evaluation set increased, save the model as new best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        epoch_time_elapsed = time.time() - epoch_start_time
        print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time_elapsed // 60, epoch_time_elapsed % 60))    
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)

    # Plot graphs
    if plot_acc:
        fig, ax = plt.subplots()
        ax.plot(train_acc_hist, 'r', label="Average training acurracy")
        ax.plot(eval_acc_hist, 'b', label="Average evaluation accuracy")
        ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Accuracy")
        ax.legend()
        plt.show()
    
    if plot_loss:
        fig, ax = plt.subplots()
        ax.plot(train_loss_hist, 'r', label="Average training loss")
        ax.plot(eval_loss_hist, 'b', label="Average evaluation loss")
        ax.set(xlabel="Epoch", ylabel="Accuracy", title="Traning Loss")
        ax.legend()
        plt.show()
    
    return model