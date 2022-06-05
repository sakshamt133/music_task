from torch.utils.data import DataLoader
from read_data import Music


music = Music()

# ================ We divide the data into a batch of 2 ============
train_batch = DataLoader(
    dataset=music,
    batch_size=4,
    shuffle=False
)
