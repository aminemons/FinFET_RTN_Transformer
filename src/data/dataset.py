import torch
from torch.utils.data import Dataset, DataLoader

class FinFETRTNDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Input tensor: [Channels, Sequence_Length]
        x = torch.from_numpy(sample['noisy_signal']).unsqueeze(0)
        
        # Target Head 1: Discrete state sequence
        y_seq = torch.from_numpy(sample['clean_signal']).long()
        
        # Target Head 2: Physical parameter regression
        y_params = torch.tensor([sample['tau_c'], sample['tau_e']], dtype=torch.float32)
        
        return x, y_seq, y_params

def create_rtn_dataloader(generator, num_samples: int, batch_size: int, num_workers: int = 32):
    print(f"Generating {num_samples} samples across {num_workers} CPU cores...")
    data_list = generator.generate_batch_multiprocess(num_samples, num_workers=num_workers)
    
    dataset = FinFETRTNDataset(data_list)
    
    # RTX A5000 Optimization: pin_memory enables fast PCIe transfers
    # drop_last=True ensures constant batch size for torch.compile()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return dataloader
