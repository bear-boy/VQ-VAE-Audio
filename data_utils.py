from torch.utils.data import Dataset, DataLoader

class TrainSet(Dataset):
    def __init__(self):
        super(TrainSet, self).__init__()

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.file_list)
