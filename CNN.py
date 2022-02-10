import statistics
import torch
import matplotlib.pyplot as plt
import time
import copy
import random
import numpy as np

def createFolds(dataset, fold_count):
    shuffled_dataset = random.sample(dataset, len(dataset)) # shuffle the list
    fold_size = len(shuffled_dataset) // fold_count # get fold size
    folds = [shuffled_dataset[i*fold_size:(i+1)*fold_size] for i in range(fold_count)] # separate in equally sized folds
    left_out = len(shuffled_dataset) % fold_count # count left out indexes
    
    # distribute left out indexes into folds
    for idx in range(left_out):
        folds[idx].append(shuffled_dataset[-1*(idx+1)])
    
    return folds

def getFoldGroups(fold_list, fold_idx):
    eval_set = [fold_set for idx,fold_set in enumerate(fold_list) if idx != fold_idx]
    return [fold_list[fold_idx], eval_set]

def processInput(input_data, input_labels, model, error_criterion):
    # Transfer inputs to GPU if available
    if torch.cuda.is_available():
        input_data = input_data.cuda()
        input_labels = input_labels.cuda()
        
    # Process the input
    output = model(input_data)
    # Get it's predictions                
    _, predictions = torch.max(output, 1)
    # Calculate batch loss
    loss_function = error_criterion(output, input_labels)

    # Get correct predictions
    _,correct_labels = torch.max(input_labels,1)

    # Get epoch global loss, count correct predictions and get accuracy
    correct_evals = torch.sum(predictions == correct_labels)
    loss = loss_function.item() / len(input_data)
    accuracy = float(correct_evals) / len(input_data)

    torch.cuda.empty_cache()

    return [loss, accuracy, loss_function]


def trainCrossValidation(model, dataset, k:int, error_criterion, optmization_algorithm, epochs:int, plot_acc = False, plot_loss = False, datasetOnGpu = False):
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
        datasetOnGpu(bool): If the input is already loaded on the GPU

    Output:
        Trained model with the weights that got the highest accuracy in the evaluation while training
    """
    # Gets execution start timestamp
    since = time.time()
    
    #Start epoch history track
    train_acc_hist = np.zeros((epochs, k))
    train_loss_hist = np.zeros((epochs, k))
    eval_acc_hist = np.zeros((epochs, k))
    eval_loss_hist = np.zeros((epochs, k))

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

    # Separate data in folds
    folds = createFolds(img_idx_list, k)

    for current_fold_idx in range(k):
        fold_start_time = time.time()
        fold_best_acc = 0.0

        # Get training and eval indexes
        [train_idxs, eval_idxs] = getFoldGroups(folds, current_fold_idx) 
        
        # Start epoch iterations
        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training phase ------------------
            model.train()  # Set model to training mode
            # Get inputs for phase
            input_data = torch.stack([dataset[i] for i in train_idxs])
            input_labels = torch.stack([dataset.getExpected(i) for i in train_idxs])

            # zero the parameter gradients
            optmization_algorithm.zero_grad()
            # Process input data
            output_loss, output_accuracy, loss_info = processInput(input_data, input_labels, model, error_criterion)
            
            # apply backward and optimize to adust weights
            loss_info.backward()
            optmization_algorithm.step()

            # Save data to training history
            train_acc_hist[epoch][current_fold_idx] = output_accuracy
            train_loss_hist[epoch][current_fold_idx] = output_loss

            # print('Training - Loss: {:.4f} Acc: {:.4f}'.format(output_loss, output_accuracy))

            # -----------------------------------

            # Evaluation phase ------------------
            model.eval()   # Set model to evaluate mode

            # Evaluation phase processes the evaluation set by folds to
            # improve memory usage. At the end the evaluation accuracy and
            # loss are retrieved with the means of the values of each 
            # evaluation fold
            eval_sets_accuracies = []
            eval_sets_losses = []
            for eval_idx_set in eval_idxs:
                input_data = torch.stack([dataset[i] for i in eval_idx_set])
                input_labels = torch.stack([dataset.getExpected(i) for i in eval_idx_set])

                # zero the parameter gradients
                optmization_algorithm.zero_grad()

                # Process input data
                output_loss, output_accuracy, _ = processInput(input_data, input_labels, model, error_criterion)
            
                # Save values
                eval_sets_accuracies.append(output_accuracy)
                eval_sets_losses.append(output_loss)

            eval_accuracy = statistics.mean(eval_sets_accuracies)
            eval_loss = statistics.mean(eval_sets_losses)

            if eval_accuracy > fold_best_acc:
                fold_best_acc = eval_accuracy
            if eval_accuracy > best_acc:
                best_acc = eval_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())

            # Save data to history
            eval_acc_hist[epoch][current_fold_idx] = eval_accuracy
            eval_loss_hist[epoch][current_fold_idx] = eval_loss

            # print('Evaluation - Loss: {:.4f} Acc: {:.4f}'.format(eval_loss, eval_accuracy))            
            # -----------------------------------

            epoch_time_elapsed = time.time() - epoch_start_time
            print('Fold {}/{} - Epoch {}/{} (elapsed time: {:.0f}m {:.0f}s)'.format(current_fold_idx +1, k, epoch + 1, epochs, epoch_time_elapsed // 60, epoch_time_elapsed % 60))

        # reset model weights
        model.load_state_dict(untrained_model_weights)
        
        time_elapsed = time.time() - fold_start_time
        print()
        print('Fold {} Training complete in {:.0f}m {:.0f}s'.format(current_fold_idx+1, time_elapsed // 60, time_elapsed % 60))
        print('Fold {} Best val Acc: {:4f}'.format(current_fold_idx+1, fold_best_acc))
        print('-' * 10)


    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # Get average curves
    avg_train_acc = np.zeros(epochs)
    avg_train_loss = np.zeros(epochs)
    avg_eval_acc = np.zeros(epochs)
    avg_eval_loss= np.zeros(epochs)
    for epoch in range(epochs):
        avg_train_acc[epoch] = statistics.mean(train_acc_hist[epoch])
        avg_train_loss[epoch] = statistics.mean(train_loss_hist[epoch])
        avg_eval_acc[epoch] = statistics.mean(eval_acc_hist[epoch])
        avg_eval_loss[epoch] = statistics.mean(eval_loss_hist[epoch])

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
