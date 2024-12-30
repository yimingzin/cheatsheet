import torch
import argparse
from plot.plot_loss_curves import plot_loss_curves
from train import train
from model.model import mnist_train_dataloader, mnist_test_dataloader, model_mnist

def get_args():
    parser = argparse.ArgumentParser(description="Train the Neural Network on images classifier task.")
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=2, help="Number of epochs")
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', '-l', metavar='LR', type=float, default=1e-3, help='learning rate', dest='lr')
    
    return parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
args = get_args()
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_mnist.parameters(), lr=args.lr)

epoch_results = train(
    epochs=args.epochs,
    model=model_mnist,
    train_dataloader=mnist_train_dataloader,
    test_dataloader=mnist_test_dataloader,
    loss_func=loss_func,
    optimizer=optimizer,
    device=device
)

plot_loss_curves(epoch_results)