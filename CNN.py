import statistics
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import copy
import random
import numpy as np
import logging

def createLogger(name: str, log_to_file = False):
    logger = logging.getLogger(name)
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(message)s')
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)

    if(log_to_file):
        f_handler = logging.FileHandler(name + ".log")
        f_handler.setLevel(logging.DEBUG)
        f_format = logging.Formatter('%(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger

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

def errorUsesClassForLabel(error_criterion):
    return type(error_criterion) == nn.CrossEntropyLoss

def processInput(input_data, input_labels, model, error_criterion):
    # Transfer inputs to GPU if available
    if torch.cuda.is_available():
        input_data = input_data.cuda()
        input_labels = input_labels.cuda()
        
    # Process the input
    output = model(input_data)
    # Get it's predictions                
    _, predictions = torch.max(output, 1)    
    # Get correct predictions
    _,correct_labels = torch.max(input_labels,1)

    # Calculate batch loss
    if(errorUsesClassForLabel(error_criterion)):
        loss_labels = torch.tensor([label.tolist().index(1) for label in input_labels], dtype=torch.long)
        if torch.cuda.is_available():
            loss_labels = loss_labels.cuda()
    else:
        loss_labels = input_labels

    loss_function = error_criterion(output, loss_labels)

    # Get epoch global loss, count correct predictions and get accuracy
    correct_evals = torch.sum(predictions == correct_labels)
    loss = loss_function.item() / len(input_data)
    accuracy = float(correct_evals) / len(input_data)

    torch.cuda.empty_cache()

    return [loss, accuracy, loss_function]

def trainCrossValidation(model, dataset, k:int, epochs:int, learning_rate= 0.1, learning_rate_drop = 0, learning_rate_drop_step_size = 0, error_criterion = nn.MSELoss(), plot_acc = False, plot_loss = False, optimization_method = optim.SGD, log_name:str = None):
    """ Trains a torchvision model using k-folds cross-validation.

    Args:
        model(torchvision.models): Neural network model.
        dataset(Dataset.ImageDataset): Dataset to train.
        k(int): Number of folds, must be at least 2.
        epochs(int): Number of training iteration for each fold.
        learning_rate(float): Learning rate for each epoch
        learning_rate_drop(float): Amount to decease the leraning rate at a given interval
        learning_rate_drop_step_size(int): Number of epochs set as the interval to decrease the learning rate
        error_criterion(torch.nn.modules.loss): Error function for the training. Uses MSE as default.
        optimization_method(torch.optim): Optimization method applied. Uses SGD as default.
        plot_acc(bool): Plot average epoch accuracy for training and evaluation after training is complete.
        plot_loss(bool): Plot average epoch loss for training and evaluation after training is complete.
        log_name(str): Filename for the log where the training proccess will be registered. If none given, no log will be created.

    Output:
        Trained model with the weights that got the highest accuracy in the evaluation while training
    """

    if(log_name != None):
        logger = createLogger(log_name, True)
    else:
        logger = createLogger("cnn_training_log")        
    logger.setLevel(logging.DEBUG)
    
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
    learning_rate_drop_function = None


    # Starts best model and accuracy to current values
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.Infinity
    
    img_count = len(dataset) # get image count
    img_idx_list = [i for i in range(img_count)] # create list with image indexes

    # Separate data in folds
    folds = createFolds(img_idx_list, k)

    for current_fold_idx in range(k):
        logger.debug(f"\nTraning fold {current_fold_idx+1}")
        optmization_algorithm = optimization_method(model.parameters(), lr=learning_rate, weight_decay=0)

        if(learning_rate_drop_step_size > 0 and learning_rate_drop != 0):
            learning_rate_drop_function = optim.lr_scheduler.StepLR(optmization_algorithm, step_size=learning_rate_drop_step_size, gamma=learning_rate_drop) 

        fold_start_time = time.time()
        fold_best_acc = 0.0

        # Get training and eval indexes
        [train_idxs, eval_idxs] = getFoldGroups(folds, current_fold_idx) 
        
        # Start epoch iterations
        for epoch in range(epochs):
            logger.debug(f"\n  Epoch {epoch+1}")
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
            if(learning_rate_drop_function != None):
                learning_rate_drop_function.step()

            # Save data to training history
            train_acc_hist[epoch][current_fold_idx] = output_accuracy
            train_loss_hist[epoch][current_fold_idx] = output_loss


            logger.debug('    Training - Loss: {:.4f} Acc: {:.4f}'.format(output_loss, output_accuracy))

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
                best_loss = eval_loss
                best_model_weights = copy.deepcopy(model.state_dict())

            # Save data to history
            eval_acc_hist[epoch][current_fold_idx] = eval_accuracy
            eval_loss_hist[epoch][current_fold_idx] = eval_loss

            logger.debug('    Evaluation - Loss: {:.4f} Acc: {:.4f}'.format(eval_loss, eval_accuracy))            
            # -----------------------------------

            epoch_time_elapsed = time.time() - epoch_start_time
            logger.info('    Fold {}/{} - Epoch {}/{} (elapsed time: {:.0f}m {:.0f}s)'.format(current_fold_idx +1, k, epoch + 1, epochs, epoch_time_elapsed // 60, epoch_time_elapsed % 60))

        # reset model weights
        model.load_state_dict(untrained_model_weights)

        
        time_elapsed = time.time() - fold_start_time
        print()
        logger.info('\n  Fold {} Training complete in {:.0f}m {:.0f}s'.format(current_fold_idx+1, time_elapsed // 60, time_elapsed % 60))
        logger.info('  Fold {} Best val Acc: {:4f}'.format(current_fold_idx+1, fold_best_acc))
        print('-' * 10)


    print()
    time_elapsed = time.time() - since
    logger.info('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
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

    logger.debug("\nSUMMARY:\n")
    logger.info('Best accuracy: {:4f}'.format(best_acc))
    logger.info('Best loss: {:4f}\n'.format(best_loss))
    logger.debug(f"Avg. training loss per epoch:\n{avg_train_loss}\n")
    logger.debug(f"Avg. training accuracy per epoch:\n{avg_train_acc}\n")
    logger.debug(f"\nAvg. evaluation loss per epoch:\n{avg_eval_loss}\n")
    logger.debug(f"Avg. evaluation accuracy per epoch:\n{avg_eval_acc}\n")

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
