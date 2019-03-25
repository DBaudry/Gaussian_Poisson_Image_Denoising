"""
training for CNN denoising

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

def check_accuracy(model, loss_fn, dataloader):
    """
    Auxiliary function that computes mean of the loss_fn
    over the dataset given by dataloader.

    Args:
        - model: a network
        - loss_fn: loss function
        - dataloader: the validation data loader

    Returns:
        - loss over the validation set
    """
    import torch

    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        model = model.cuda()
        try:
            loss_fn = loss_fn.cuda()
        except:
            pass
        dtype   = torch.cuda.FloatTensor

    loss = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for (x, y) in dataloader:

            # transform mini-batch to tensors
            x_var = x.type(dtype)
            y_var = y.type(dtype)

            # apply model to x mini-batch
            out   = model(x_var)

            # accumulate loss
            loss += loss_fn(out, y_var)

    # return loss divided by number of mini-batches
    return loss/len(dataloader)


def trainmodel(model, loss_fn, loader_train, loader_val=None,
               optimizer=None, scheduler=None, num_epochs = 1,
               learning_rate=0.001, weight_decay=0.0, loss_every=10,
               save_every=10, filename=None, val_loss_fn=None):
    """
    function that trains a network model
    Args:
        - model       : network to be trained
        - loss_fn     : loss functions
        - loader_train: dataloader for the training set
        - loader_val  : dataloader for the validation set (default None)
        - optimizer   : the gradient descent method (default None)
        - scheduler   : handles the hyperparameters of the optimizer
        - num_epoch   : number of training epochs
        - save_every  : save the model every n epochs
        - filename    : base filename for the saved models
        - loss_every  : print the loss every n epochs
        - learning_rate: learning rate (default 0.001)
        - weight_decay: weight decay regularization (default 0.0)
        - val_loss_fn : if set, indicates validation loss function, otherwise uses loss_fn 
    Returns:
        - model          : trained network
        - loss_history   : history of loss values on the training set
        - valloss_history: history of loss values on the validation set
    """
    import torch
    from time import time
    import numpy as np

    # if not set validation loss is loss_fn
    if val_loss_fn == None:
        val_loss_fn = loss_fn
        
    dtype = torch.FloatTensor
    # GPU
    if torch.cuda.is_available():
        model   = model.cuda()
        try:
            loss_fn = loss_fn.cuda()
            val_loss_fn = val_loss_fn.cuda()
        except:
            pass
        dtype   = torch.cuda.FloatTensor

    if optimizer == None or scheduler == None:
        # Default optimizer and scheduler

        # The optimizer is in charge of updating the parameters
        # of the model. It has hyper-parameters for controlling
        # the gradient update, such as the learning rate (lr) and
        # the regularization such as the weight_decay
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=weight_decay, amsgrad=False)

        # The learning rate scheduler monitors the evolution of the loss
        # and adapts the learning rate to avoid plateaus. We will use
        # a scheduler available in torch that reduces the lr by 'factor'
        # if in the last epochs there hasn't been a significant
        # reduction of the validation loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                            mode='min', factor=0.5, patience=50,
                            mode='min', factor=0.8, patience=50,
                            verbose=True, threshold=0.0001,
                            threshold_mode='rel', cooldown=0,
                            min_lr=0, eps=1e-08)

    loss_history=[]
    valloss_history=[]
    
    # Display initial training and validation loss
    message=''
    if loader_val is not None:
        valloss = check_accuracy(model, val_loss_fn, loader_val)
        message = ', val_loss = %.4f' % valloss.item()

    print('Epoch %5d/%5d, ' % (0, num_epochs) +
          'loss = %.4f%s'% (-1, message))

    # Save initial results
    if filename:
        torch.save([model, optimizer, loss_history, valloss_history],
                   filename+'%04d.pt' % 0)

    # Main training loop
    for epoch in range(num_epochs):

        # The data loader iterates once over the whole data set
        for (x, y) in loader_train:
            # make sure that the models is in train mode
            model.train()

            # Apply forward model and compute loss on the batch
            x = x.type(dtype) # Convert data into pytorch 'variables'
            y = y.type(dtype) # for computing the backprop of the loss
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # Zero out the gradients of parameters that the optimizer
            # will update. The optimizer is already linked to the
            # network parameters.
            optimizer.zero_grad()

            # Backwards pass: compute the gradient of the loss with
            # respect to all the learnable parameters of the model.
            loss.backward()

            # Update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        # Store loss history to plot it later
        loss_history.append(loss)
        if loader_val is not None:
            valloss = check_accuracy(model, val_loss_fn, loader_val)
            valloss_history.append(valloss)

        # Display current loss and compute validation loss
        if ((epoch + 1) % loss_every == 0):
            message=''
            if loader_val is not None:
                message = ', val_loss = %.4f' % valloss.item()

            print('Epoch %5d/%5d, ' % (epoch + 1, num_epochs) +
                  'loss = %.4f%s'% (loss.item(), message))

        # Save partial results
        if filename and ((epoch + 1) % save_every == 0):
            torch.save([model, optimizer, loss_history, valloss_history],
                       filename+'%04d.pt' % (epoch + 1))
            print('Epoch %5d/%5d, checkpoint saved' % (epoch + 1, num_epochs))

        # scheduler update
        scheduler.step(loss.data)

    # Save last result
    if filename:
        torch.save([model, optimizer, loss_history, valloss_history],
                    filename+'%04d.pt' % (epoch + 1))

    return model, loss_history, valloss_history
