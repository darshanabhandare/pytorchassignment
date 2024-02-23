import torch
import matplotlib.pyplot as plt


def is_cuda_available() -> bool:
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    return cuda

def plot_data(train_loader):
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()