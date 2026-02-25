# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model.

## DESIGN STEPS
## STEP 1:
Import required libraries.Then dataset is loaded and define the training and testing dataset.

## STEP 2:
initialize the model,loss function,optimizer. CrossEntropyLoss for multi-class classification and Adam optimizer for efficient training.

## STEP 3:
Train the model with training dataset.

## STEP 4:
Evaluate the model with testing dataset.

## STEP 5:
Make Predictions on New Data.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match the dataset classes

num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)



# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

    # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: YUVASHREE R")
    print("Register Number: 212224040378")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="720" height="771" alt="image" src="https://github.com/user-attachments/assets/24afbf2a-affc-48c4-9091-fa206c6be6ef" />


### Confusion Matrix

<img width="642" height="595" alt="image" src="https://github.com/user-attachments/assets/3b8f9942-3655-4596-a500-7fac7280a872" />


### Classification Report


<img width="362" height="185" alt="image" src="https://github.com/user-attachments/assets/969f46ef-3a88-41c9-aed1-9e4292025c92" />




### New Sample Prediction
<img width="415" height="827" alt="image" src="https://github.com/user-attachments/assets/ad8113d6-08eb-4ed7-b937-5ee43804f256" />


## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
