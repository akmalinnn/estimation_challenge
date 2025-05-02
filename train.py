import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os


# def evaluate(model, data, flag, device):
#     criterion = torch.nn.MSELoss()
    
#     if flag:
#         model.eval()
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs).squeeze()
#         loss = criterion(outputs, labels)
        
#     if not flag:
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         with torch.no_grad():
#             outputs = model(inputs).squeeze()
#             loss = criterion(outputs, labels)
#             loss.backward()
#     return loss


def train(model, epochs, train_loader, val_loader, optimizer,
          RESULTS_PATH, MODEL_PATH=None, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.MSELoss()

    plot_dir = os.path.join(os.path.dirname(RESULTS_PATH), 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    results_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])
    results_df.set_index('epoch', inplace=True)

    print("Starting training...")
    for epoch in range(epochs):

        # For Training
        model.train()
        epoch_train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()
            print ("out: ", outputs)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()

            if i % 50 == 49:
                avg_loss = epoch_train_loss / (i + 1)
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # For Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print losses per epoch
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

         # For testing
         #
         #
        
        if scheduler:
            scheduler.step(avg_val_loss)


        # Save results
        results_df.loc[epoch] = [avg_train_loss, avg_val_loss]
        results_df.to_csv(RESULTS_PATH)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if MODEL_PATH:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f'New best model saved at epoch {epoch+1} with val loss: {best_val_loss:.4f}')

        # Save plot every 10 epochs
        if (epoch+1) % 10 == 0:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(results_df.index, results_df['train_loss'], label='Training Loss')
            plt.plot(results_df.index, results_df['val_loss'], label='Validation Loss')
            plt.title('Loss Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'training_progress_epoch_{epoch+1:04d}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved training plot to {plot_path}')

    print('Training completed')
    return model
