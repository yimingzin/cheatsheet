import matplotlib.pyplot as plt

def plot_loss_curves(results):
    """Plots training curves of a results dictionary

    Args:
        results (dict): dictionary containing list of values e.g.
        {
            "train_loss": [...],
            "train_acc": [...],
            "test_loss": [...],
            "test_acc": [...]
            
        }
    """
    
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]
    
    train_acc = results["train_acc"]
    test_acc = results["test_acc"]
    
    epochs = range(len(train_loss))
    
    # Plot loss
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, test_acc, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.show()