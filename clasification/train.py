import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def train_classifier(model, epochs, train_loader, val_loader, test_loader,optimizer,
    MODEL_PATH=None, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.BCELoss() 

    plot_dir = MODEL_PATH
    os.makedirs(plot_dir, exist_ok=True)
    metrics_csv_path = os.path.join(plot_dir, 'classifier_metrics.csv')

    with open(metrics_csv_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,test_loss,avg_train_error,avg_val_error,test_accuracy\n")
    
    train_losses = []
    val_losses = []
    test_losses = []
    avg_train_error_plot = []
    avg_val_error_plot = []
    test_accuracies = []
    

    best_val_loss = float('inf')
    
    results_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    results_df.set_index('epoch', inplace=True)

    print("classification training...")
    for epoch in range(epochs):

        # Training phase
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        lendata = 0
        cumulative_error = 0.0

        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100):

            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(dim=1)

            # print (outputs, labels)
            lendata += 8 #len(outputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            

            

            # print(outputs)
            # print(labels)
            # batch_error = 0
            # batch_error += torch.abs(labels - outputs).sum().item()
            
            # print("\nOutput):", outputs)
            # print("label):", labels)
            
            # print("Error:", torch.abs(labels - outputs))
            # print(f"error: {batch_error/batch_size :.4f}")
            epoch_train_loss += loss.item()
            cumulative_error += torch.abs(labels - outputs).sum().item()
            # correct_train += (outputs == labels).sum().item()

        
        avg_train_loss = epoch_train_loss / lendata
        avg_train_error = cumulative_error / lendata

        train_losses.append(avg_train_loss)
        avg_train_error_plot.append(avg_train_error)
        print(f'Avg Train Loss: {avg_train_loss:.4f}, Avg Train Error: {avg_train_error:.4f}')

      
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        lendata_val = 0
        val_cumulative_error = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze(dim=1)
                lendata_val += 8


                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_cumulative_error += torch.abs(labels - outputs).sum().item()

                # correct_val += (outputs == labels).sum().item()
                

        avg_val_loss = epoch_val_loss / lendata_val
        avg_val_error = val_cumulative_error / lendata_val


        val_losses.append(avg_val_loss)
        avg_val_error_plot.append(avg_val_error)
        print(f'Avg Val Loss: {avg_val_loss:.4f}, Avg Val Error: {avg_val_error:.4f}')



        # Test phase
        model.eval()
        epoch_test_loss = 0.0
        correct_test = 0
        lendata_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs).squeeze(dim=1)

                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()

                predicted = (outputs >= 0.5).float()
                correct_test += (predicted == labels).sum().item()
                lendata_test += 8 

        avg_test_loss = epoch_test_loss / lendata_test  
        test_accuracy = correct_test / lendata_test

        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

        if scheduler:
            scheduler.step(avg_val_loss)
            # scheduler.step()

        with open(metrics_csv_path, 'a') as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{avg_test_loss:.6f},"
                    f"{avg_train_error:.6f},{avg_val_error:.6f},{test_accuracy:.6f}\n")
        
        if MODEL_PATH:
            epoch_model_path = os.path.join(os.path.dirname(MODEL_PATH), f'model_CLF_epoch_{epoch+1:04d}.pt')
            torch.save(model.state_dict(), epoch_model_path)
            print(f'Saved model for epoch {epoch+1} to {epoch_model_path}')

        # Plotting
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            plt.figure(figsize=(20, 5))  # Adjusted size for 3 plots in 1 row

            # 1st plot: Loss
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # 2nd plot: Error
            plt.subplot(1, 3, 2)
            plt.plot(avg_train_error_plot, label='Training Error')
            plt.plot(avg_val_error_plot, label='Validation Error')
            plt.title('Error')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.legend()

            # 3rd plot: Accuracy
            plt.subplot(1, 3, 3)
            plt.plot(test_accuracies, label='Test Accuracy')
            plt.title('Accuracy Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'classifier_progress_epoch_{epoch+1:04d}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved training plot to {plot_path}')

    print('Classification training completed')


    return model

#BCE LOSS 

# def train_classifier(model, epochs, train_loader, val_loader, optimizer,
#                    RESULTS_PATH, MODEL_PATH=None, scheduler=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     criterion = torch.nn.BCELoss()

#     plot_dir = os.path.join(os.path.dirname(RESULTS_PATH), 'training_plots')
#     os.makedirs(plot_dir, exist_ok=True)
    
#     train_losses = []
#     val_losses = []
#     train_accuracies = []
#     val_accuracies = []
#     best_val_loss = float('inf')
    
#     results_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
#     results_df.set_index('epoch', inplace=True)

#     print("classification training...")
#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         epoch_train_loss = 0.0
#         correct_train = 0
#         total_train = 0
        
#         for i, (inputs, labels) in enumerate(train_loader):
            
#             inputs, labels = inputs.to(device), labels.to(device).float()
#             outputs = model(inputs).squeeze(dim=1)
            
#             loss = criterion(outputs, labels)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_train_loss += loss.item()
#             predicted = (outputs > 0.5).float()
#             # print (outputs)
#             # print (labels)
#             # print (predicted)
#             correct_train += (predicted == labels).sum().item()
#             total_train += labels.size(0)

#             if i % 50 == 49:
#                 avg_loss = epoch_train_loss / (i + 1)
#                 batch_acc = correct_train / total_train
#                 print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Acc: {batch_acc:.4f}')

#         avg_train_loss = epoch_train_loss / len(train_loader)
#         train_accuracy = correct_train / total_train
#         train_losses.append(avg_train_loss)
#         train_accuracies.append(train_accuracy)

#         # Validation phase
#         model.eval()
#         epoch_val_loss = 0.0
#         correct_val = 0
#         total_val = 0
        
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device).float()
#                 outputs = model(inputs).squeeze(dim=1)
#                 loss = criterion(outputs, labels)
#                 epoch_val_loss += loss.item()
                
#                 predicted = (outputs > 0.5).float()
#                 correct_val += (predicted == labels).sum().item()
#                 total_val += labels.size(0)
        
#         avg_val_loss = epoch_val_loss / len(val_loader)
#         val_accuracy = correct_val / total_val
#         val_losses.append(avg_val_loss)
#         val_accuracies.append(val_accuracy)

#         print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
#         print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

#         if scheduler:
#             scheduler.step(avg_val_loss)

#         # Save results
#         results_df.loc[epoch] = [avg_train_loss, avg_val_loss, train_accuracy, val_accuracy]
#         results_df.to_csv(RESULTS_PATH)
        
#         if MODEL_PATH:
#             epoch_model_path = os.path.join(os.path.dirname(MODEL_PATH), f'model_CLF_epoch_{epoch+1:04d}.pt')
#             torch.save(model.state_dict(), epoch_model_path)
#             print(f'Saved model for epoch {epoch+1} to {epoch_model_path}')

#         # Plotting
#         if (epoch+1) % 10 == 0 or epoch == epochs-1:
#             plt.figure(figsize=(15, 5))
            
#             plt.subplot(1, 2, 1)
#             plt.plot(train_losses, label='Training Loss')
#             plt.plot(val_losses, label='Validation Loss')
#             plt.title('Loss Curves')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.legend()
            
#             plt.subplot(1, 2, 2)
#             plt.plot(train_accuracies, label='Training Accuracy')
#             plt.plot(val_accuracies, label='Validation Accuracy')
#             plt.title('Accuracy Curves')
#             plt.xlabel('Epoch')
#             plt.ylabel('Accuracy')
#             plt.legend()
            
#             plt.tight_layout()
#             plot_path = os.path.join(plot_dir, f'classifier_progress_epoch_{epoch+1:04d}.png')
#             plt.savefig(plot_path)
#             plt.close()
#             print(f'Saved training plot to {plot_path}')

#     print('Classification training completed')
#     return model