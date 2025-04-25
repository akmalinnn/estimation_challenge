import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

# def evaluate(model, data_loader, device):
#     """
#     Calculate classification error (%) for given model
#     and data set.
    
#     Parameters:
    
#     - model: A Trained Pytorch Model 
#     - data_loader: A Pytorch data loader object
#     """
    
#     y_true = np.array([], dtype=np.int)
#     y_pred = np.array([], dtype=np.int)
    
#     with torch.no_grad():
#         for data in data_loader:
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
            
#             y_true = np.concatenate((y_true, labels.cpu()))
#             y_pred = np.concatenate((y_pred, predicted.cpu()))
    
#     error = np.sum(y_pred != y_true) / len(y_true)
#     return error

def evaluate(model, data_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(data_loader.dataset)


# def train(model, epochs, train_loader, test_loader, criterion, 
#           optimizer, RESULTS_PATH, scheduler=None, MODEL_PATH=None):
#     """
#     End to end training as described by the original resnet paper:
#     https://arxiv.org/abs/1512.03385
    
#     Parameters
#     ----------------
    
#     - model: The PyTorch model to be trained
#     - n:   Determines depth of the neural network 
#            as described in paper
#     - train_loader: 
#            PyTorch dataloader object for training set
#     - test_loader: 
#            PyTorch dataloader object for test set
#     """
    
#     # Run on GPU if available
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
#     model.to(device)
    
#     # Training loop
#     # -------------------------------
#     cols       = ['epoch', 'train_loss', 'train_err', 'test_err']
#     results_df = pd.DataFrame(columns=cols).set_index('epoch')
#     print('Epoch \tBatch \tNLLLoss_Train')
    
#     for epoch in range(epochs):  # loop over the dataset multiple times
        
#         model.train()
#         running_loss  = 0.0
#         best_test_err = 1.0
#         for i, data in enumerate(train_loader, 0):   # Do a batch iteration
            
#             # get the inputs
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             # zero the parameter gradients
#             optimizer.zero_grad()
            
#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             # print average loss for last 50 mini-batches
#             running_loss += loss.item()
#             if i % 50 == 49:
#                 print('%d \t%d \t%.3f' %
#                       (epoch + 1, i + 1, running_loss / 50))
#                 running_loss = 0.0
        
#         if scheduler:
#             scheduler.step()
        
#         # Record metrics
#         model.eval()
#         train_loss = loss.item()
#         train_err = evaluate(model, train_loader, device)
#         test_err = evaluate(model, test_loader, device)
#         results_df.loc[epoch] = [train_loss, train_err, test_err]
#         results_df.to_csv(RESULTS_PATH)
#         print(f'train_err: {train_err} test_err: {test_err}')
        
#         # Save best model
#         if MODEL_PATH and (test_err < best_test_err):
#             torch.save(model.state_dict(), MODEL_PATH)
#             best_test_err = test_err
        
        
    
#     print('Finished Training')
#     model.eval()
#     return model

def train(model, epochs, train_loader, test_loader, criterion,
          optimizer, RESULTS_PATH, scheduler=None, MODEL_PATH=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create directory for plots if it doesn't exist
    plot_dir = os.path.join(os.path.dirname(RESULTS_PATH), 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)

    cols = ['epoch', 'train_loss', 'train_err', 'test_err']
    results_df = pd.DataFrame(columns=cols).set_index('epoch')
    print('Epoch \tBatch \tLoss_Train')

    best_test_err = float('inf')
    
    # Initialize lists to store metrics for plotting
    train_losses = []
    train_errors = []
    test_errors = []
    epochs_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print('%d \t%d \t%.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

        model.eval()
        train_loss = loss.item()
        train_err = evaluate(model, train_loader, device)
        test_err = evaluate(model, test_loader, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_errors.append(train_err)
        test_errors.append(test_err)
        epochs_list.append(epoch + 1)
        
        results_df.loc[epoch] = [train_loss, train_err, test_err]
        results_df.to_csv(RESULTS_PATH)
        print(f'train_err: {train_err} test_err: {test_err}')

        if scheduler:
            scheduler.step(test_err)

        # Save plots every 10 epochs
        if (epoch + 1) % 10 == 0 :
            plt.figure(figsize=(12, 4))
            
            # Plot Training and Test Error
            plt.subplot(1, 2, 1)
            plt.plot(epochs_list, train_errors, label='Training Error')
            plt.plot(epochs_list, test_errors, label='Test Error')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.title(f'Training Progress (Epoch {epoch+1})')
            plt.legend()
            plt.grid(True)
            
            # Plot Training Loss
            plt.subplot(1, 2, 2)
            plt.plot(epochs_list, train_losses, label='Training Loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss (Epoch {epoch+1})')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'training_epoch_{epoch+1:04d}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved training plot to {plot_path}')

        if epoch % 100 == 99:
            if MODEL_PATH:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Model saved at epoch {epoch + 1}")

        if MODEL_PATH and (test_err < best_test_err):
            torch.save(model.state_dict(), MODEL_PATH)
            best_test_err = test_err

    print('Finished Training')
    model.eval()
    return model
