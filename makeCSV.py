import torch
import pandas as pd
import argparse
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class TestImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        # return image path
        return super(TestImageFolder, self).__getitem__(index), self.imgs[index][0].split('/')[-1]


def eval():
    ########## You can change this part only in this cell ##########
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    ################################################################

    test_dataset = TestImageFolder('./test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=4, shuffle=False)

    model = Model()
    model = model.cuda()
    model.load_state_dict(torch.load(SAVEPATH+'model_weight.pth'))
    model.eval()

    print('Make an evaluation csv file for kaggle submission...')
    Category = []
    Id = []
    for data in test_loader:
        (input, _), name = data
        
        input = input.cuda()
        output = model(input)
        output = torch.argmax(output, dim=1)
        Id = Id + list(name)
        Category = Category + output.tolist()

    #Id = list(range(0, 90000))
    samples = {
       'Id': Id,
       'Target': Category 
    }
    df = pd.DataFrame(samples, columns=['Id', 'Target'])

    df.to_csv(SAVEPATH+'submission.csv', index=False)
    print('Done!!')


if __name__ == "__main__":
    eval()
