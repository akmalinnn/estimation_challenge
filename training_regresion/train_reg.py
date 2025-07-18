import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def train_reg(model, epochs, train_loader, val_loader, test_loader, optimizer
    , MODEL_PATH=None, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.MSELoss()

    plot_dir = MODEL_PATH
    os.makedirs(plot_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    
    results_df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'test_loss'])
    results_df.set_index('epoch', inplace=True)

    print("Starting training...")
    for epoch in range(epochs):

        # For Training
        model.train()
        epoch_train_loss = 0.0
        
        # for i, (inputs, labels) in tqdm(enumerate(train_loader)):
        for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze()
            labels = labels.squeeze()
            # print ("out: ", outputs)
            # print ("label: ", labels)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()

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

       
        # For testing
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                epoch_test_loss += loss.item()
        
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        
        if scheduler:
            scheduler.step(avg_val_loss)

         # Print losses per epoch
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}")



        # Save results
        results_df.loc[epoch] = [avg_train_loss, avg_val_loss, avg_test_loss]
        results_df.to_csv("../result/training_reg/model_plot/results.csv")

        if MODEL_PATH:
            epoch_model_path = os.path.join(os.path.dirname(MODEL_PATH), f'model_REG_epoch_{epoch+1:04d}.pt')
            torch.save(model.state_dict(), epoch_model_path)
            print(f'Saved model for epoch {epoch+1} to {epoch_model_path}')
        


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
            plt.plot(results_df.index, results_df['test_loss'], label='Test Loss')
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