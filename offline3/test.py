import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import f1_score
from torchvision import datasets, transforms
import torch

# Dense Layer
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros(output_size)
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dout):
        dx = np.dot(dout, self.weights.T)
        dw = np.dot(self.inputs.T, dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db

# Batch Normalization
class BatchNormalization:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.inputs = None  # Initialize inputs to None
        self.training = True  # Default to training mode
    
    def forward(self, x):
        self.inputs = x  # Store input for use in backward pass
        if self.training:
            self.sample_mean = np.mean(x, axis=0)
            self.sample_var = np.var(x, axis=0)
            self.x_hat = (x - self.sample_mean) / np.sqrt(self.sample_var + self.epsilon)
            self.out = self.gamma * self.x_hat + self.beta
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sample_var
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.out = self.gamma * self.x_hat + self.beta
        return self.out
    
    def backward(self, dout):
        N, D = self.inputs.shape  # Get the shape of the input
        dx_hat = dout * self.gamma
        
        # Compute gradients
        dsample_var = np.sum(dx_hat * (self.x_hat * -0.5 * (self.sample_var + self.epsilon) ** -1.5), axis=0)
        dsample_mean = np.sum(dx_hat * (-1 / np.sqrt(self.sample_var + self.epsilon)), axis=0) + dsample_var * np.mean(-2 * (self.inputs - self.sample_mean), axis=0)
        dx = (dx_hat * (1 / np.sqrt(self.sample_var + self.epsilon))) + (dsample_var * 2 * (self.inputs - self.sample_mean) / N) + (dsample_mean / N)
        
        dgamma = np.sum(dout * self.x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        return dx, dgamma, dbeta

# ReLU Activation
class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(float)
        return x * self.mask
    
    def backward(self, dout):
        return dout * self.mask

# Dropout
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True  # Default to training mode
        
    def forward(self, x):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
            return x * self.mask
        else:
            return x
    
    def backward(self, dout):
        return dout * self.mask

# Softmax Cross-Entropy Loss
class SoftmaxCrossEntropyLoss:
    def forward(self, logits, labels):
        self.labels = labels
        self.probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
        self.probs /= np.sum(self.probs, axis=1, keepdims=True)
        loss = -np.mean(np.log(self.probs[range(len(labels)), labels]))
        return loss
    
    def backward(self):
        dout = self.probs.copy()
        dout[range(len(self.labels)), self.labels] -= 1
        dout /= len(self.labels)
        return dout

# Adam Optimizer
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        
    def update(self, params, grads, t):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            if grad is not None:  # Check if gradient is not None
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** t)
                v_hat = self.v[i] / (1 - self.beta2 ** t)
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Neural Network Model
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, num_classes):
        self.layers = []
        self.training = True  # Default to training mode
        self.layers.append(DenseLayer(input_size, hidden_sizes[0]))
        self.layers.append(BatchNormalization(hidden_sizes[0]))
        self.layers.append(ReLU())
        self.layers.append(Dropout(0.2))
        
        for i in range(1, len(hidden_sizes)):
            self.layers.append(DenseLayer(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(BatchNormalization(hidden_sizes[i]))
            self.layers.append(ReLU())
            self.layers.append(Dropout(0.2))
        
        self.layers.append(DenseLayer(hidden_sizes[-1], num_classes))
        self.loss = SoftmaxCrossEntropyLoss()
        self.optimizer = Adam()
        
    def forward(self, x):
        for layer in self.layers:
            layer.training = self.training  # Set training mode for each layer
            x = layer.forward(x)
        return x

    def backward(self, dout):
        grads = []
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                outputs = layer.backward(dout)
                dout = outputs[0]
                if len(outputs) > 1:
                    grads.append(outputs[1])  # Gradient for weights
                    grads.append(outputs[2])  # Gradient for biases
                else:
                    grads.append(None)  # No gradient if layer has no parameters
        return grads[::-1]  # Return in reverse order

    def train(self, X, y, val_X, val_y, epochs, batch_size):
        num_batches = len(X) // batch_size
        train_losses, val_losses, train_accs, val_accs, val_f1s = [], [], [], [], []

        for epoch in range(epochs):
            pbar = tqdm(range(num_batches), leave=False)
            self.training = True  # Enable training mode
            for i in pbar:
                batch_X = X[i*batch_size:(i+1)*batch_size]
                batch_y = y[i*batch_size:(i+1)*batch_size]

                logits = self.forward(batch_X)
                loss = self.loss.forward(logits, batch_y)
                grads = self.backward(self.loss.backward())
                # Ensure params and grads are aligned
                self.optimizer.update(self.params, grads, epoch * num_batches + i + 1)

                train_loss = loss
                train_acc = np.mean(np.argmax(logits, axis=1) == batch_y)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                pbar.set_description(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            self.training = False  # Disable training mode for validation
            val_logits = self.forward(val_X)
            val_loss = self.loss.forward(val_logits, val_y)
            val_acc = np.mean(np.argmax(val_logits, axis=1) == val_y)
            val_f1 = f1_score(val_y, np.argmax(val_logits, axis=1), average='macro')

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_f1s.append(val_f1)

            print(f"Epoch: {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        return train_losses, val_losses, train_accs, val_accs, val_f1s

    @property
    def params(self):
        return [param for layer in self.layers if hasattr(layer, 'weights') for param in [layer.weights, layer.biases]]

# Load Data (Example using FashionMNIST)
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # Flatten
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    x_train = []
    y_train = []
    for data in train_loader:
        x_train.append(data[0].numpy())
        y_train.append(data[1].numpy())
    
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_val = []
    y_val = []
    for data in test_loader:
        x_val.append(data[0].numpy())
        y_val.append(data[1].numpy())

    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)

    return x_train, y_train, x_val, y_val

def main():
    x_train, y_train, x_val, y_val = load_data()
    input_size = x_train.shape[1]
    hidden_sizes = [256, 128]  # Hidden layers sizes
    num_classes = 10  # For FashionMNIST

    neural_network = NeuralNetwork(input_size, hidden_sizes, num_classes)
    epochs = 20
    batch_size = 64

    train_losses, val_losses, train_accs, val_accs, val_f1s = neural_network.train(x_train, y_train, x_val, y_val, epochs, batch_size)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
