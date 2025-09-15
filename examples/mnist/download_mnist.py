# Manually download MNIST if the cluster's computing Nodes that lacks global etwork access
import torchvision

torchvision.datasets.MNIST(root=".data", train=True, download=True)
print('Done')