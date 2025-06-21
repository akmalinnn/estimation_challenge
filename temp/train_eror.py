import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

def evaluate(model, data, flag, device):
    criterion = torch.nn.MSELoss()
    
    if flag:
        model.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        
    if not flag:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
    return loss


def train(model, epochs, train_loader, val_loader, criterion,
          optimizer, RESULTS_PATH, scheduler=None, MODEL_PATH=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

   
    plot_dir = os.path.join(os.path.dirname(RESULTS_PATH), 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)

    cols = ['epoch', 'train_loss', 'test_loss']
    results_df = pd.DataFrame(columns=cols).set_index('epoch')
    print('Epoch \tBatch \tLoss_Train')

    best_val_err = float('inf')
    
    # Initialize lists to store metrics for plotting
    train_losses = []
    train_errors = []
    test_errors = []
    epochs_list = []

    # print (len(train_loader))
    # print (len(val_loader))

    for epoch in range(epochs):
        # model.train()
        running_loss = 0.0
        criterion = torch.nn.MSELoss()

        # for training
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            if i % 50 == 49:
                print('%d \t%d \t%.3f' %
                      (epoch + 1, i + 1, train_loss / 50))
            #     train_loss = 0.0
            # print (train_loss)
        


        # for validation
        val_loss = 0
        for i, data in enumerate(val_loader, 0):            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            # print (val_loss)
        
        if scheduler:
            scheduler.step(val_loss)

        # for testing
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        train_err = evaluate(model, train_batch, False, device)
        test_err = evaluate(model, val_batch, True, device)

        #model.eval()
        train_loss = loss.item()
        # train_err = evaluate(model, train_loader, False, device)
        # test_err = evaluate(model, val_loader, True, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_errors.append(train_err)
        test_errors.append(test_err)
        epochs_list.append(epoch + 1)
        
        results_df.loc[epoch] = [train_err.cpu().item(), test_err.cpu().item()]
        results_df.to_csv(RESULTS_PATH)
        print(f'train_err: {train_err} test_err: {test_err}')

        
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(12, 4))

            # # Plot training and validation loss
            # plt.subplot(1, 2, 1)
            # plt.plot(epochs_list, train_losses, label='Train Loss', color='blue')
            # plt.plot(epochs_list, val_losses, label='Validation Loss', color='green')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.title(f'Training and Validation Loss (Epoch {epoch+1})')
            # plt.legend()
            # plt.grid(True)

            # Plot training and validation errors
            plt.subplot(1, 2, 2)
            plt.plot(epochs_list, train_errors, label='Train loss', color='red')
            plt.plot(epochs_list, val_errors, label='Validation loss', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.title(f'Training and Validation loss (Epoch {epoch+1})')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'training_epoch_{epoch+1:04d}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved training plot to {plot_path}')

        # Save the model if validation error improves
        if epoch % 100 == 99:
            if MODEL_PATH:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Model saved at epoch {epoch + 1}")

        if MODEL_PATH and (val_loss < best_val_err):
            torch.save(model.state_dict(), MODEL_PATH)
            best_val_err = val_loss

    print('Finished Training')
    #model.eval()
    return model
