from make_batches import train_batch
from model import MyModel
import torch


# ====== Hyperparameter, Optimizer and loss Function =======

in_feats = 2
n_classes = 5
lr = 0.001
epochs = 30

model = MyModel(in_feats, n_classes)
opti = torch.optim.Adam(model.parameters(), lr=lr)
l_fun = torch.nn.CrossEntropyLoss()

# ============== Training of the Model ================

for epoch in range(epochs):
    for i, (data, labels) in enumerate(train_batch):
        y_hat = model(data.float())
        loss = l_fun(y_hat, labels)
        print(f'for epoch {epoch} batch no {i} loss is {loss}')
        loss.backward()
        opti.step()
        opti.zero_grad()


# =============== To Save the Model ===================
path = "model.pt"

checkpoint = {
    "model_state_dict": model.state_dict(),
    "opti_state_dict": opti.state_dict(),
    "loss": l_fun
}

torch.save(checkpoint, path)
