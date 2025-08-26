import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

def create_mnist_dataset(root_dir="./data", batch_size=128, train=True, download=True):
    """
    Create an MNIST dataset loader.
    
    Args:
        root_dir (str): Directory to store the dataset
        batch_size (int): Size of mini-batch
        train (bool): Whether to load the training or test set
        download (bool): Whether to download the dataset if not present
        
    Returns:
        DataLoader: PyTorch DataLoader for MNIST dataset
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Create dataset
    dataset = datasets.MNIST(
        root=root_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    # Create data loader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,  # Shuffle only if training
        num_workers=2
    )
    
    return loader

class MNISTDataset(Dataset):
    """Custom MNIST dataset that provides additional functionality if needed."""
    
    def __init__(self, root_dir="./data", train=True, transform=None, download=True):
        """
        Initialize the MNIST dataset.
        
        Args:
            root_dir (str): Directory to store the dataset
            train (bool): Whether to load the training or test set
            transform: Optional transform to be applied on a sample
            download (bool): Whether to download the dataset if not present
        """
        self.mnist = datasets.MNIST(
            root=root_dir,
            train=train,
            download=download,
            transform=transform if transform else transforms.ToTensor()
        )
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        image, label = self.mnist[idx]
        return image, label


def print_mnist_image(image, threshold=0.5):
    """
    Print an MNIST image to stdout using ASCII characters.
    
    Args:
        image (torch.Tensor): MNIST image tensor with shape [1, 28, 28]
        threshold (float): Value above which a pixel is considered 'on'
    """
    # Check if image is a tensor and convert to numpy if needed
    if isinstance(image, torch.Tensor):
        # Remove normalization if present (approximately)
        if image.min() < 0:
            image = image * 0.3081 + 0.1307
        
        # Ensure proper dimensions and get the image data
        if image.dim() == 4:  # Batch of images
            image = image[0]  # Take the first image
        if image.dim() == 3 and image.size(0) == 1:  # Single channel image
            image = image.squeeze(0)
        
        image = image.numpy()
    
    # Map pixel intensities to ASCII characters
    # Darker pixels get heavier characters
    chars = '⬛⬜'
    
    # Print the image
    for i in range(image.shape[0]):
        row = ''
        for j in range(image.shape[1]):
            # Map pixel value to character index
            pixel_value = image[i, j]
            char_idx = min(int(pixel_value * len(chars)), len(chars) - 1)
            row += chars[char_idx]
        print(row)

if __name__ == "__main__":
    # Create a dataset and data loader for testing
    train_loader = create_mnist_dataset(train=True)
    test_loader = create_mnist_dataset(train=False)

    # Iterate through the training data
    for images, labels in train_loader:
        print("Batch of images:", images.size())
        print("Batch of labels:", labels.size())

        # Print the first image in the batch
        print_mnist_image(images[0])
        print("Label:", labels[0].item())

        break  # Just show the first batch