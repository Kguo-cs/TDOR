import torch
from torch.utils.data import DataLoader


from model.full_model import Model

# Initialize device:
device = torch.device( "cuda:0")

dataset="sdd"

if dataset=="ind":
    horizon = 30
    fut_len = 30
    grid_extent = 25
    nei_dim=0
    type="test"

    from data.IND.inD import inD as DS
else:
    horizon = 20
    fut_len = 12
    grid_extent = 20
    nei_dim=2
    type="sddtest"

    from data.SDD.sdd import sdd as DS


net = Model(horizon, fut_len,nei_dim,grid_extent).float().to(device)


if dataset=="ind":
    checkpoint = torch.load("./pretrained/indend.tar",map_location='cuda:0')
elif dataset=="trajnet":
    checkpoint = torch.load("./pretrained/trajnetend.tar",map_location='cuda:0')
else:
    checkpoint = torch.load("./pretrained/sddend.tar",map_location='cuda:0')

test_set = DS(dataset,horizon=horizon, fut_len=fut_len, type="test", grid_extent=grid_extent)



test_dl = DataLoader(test_set,
                    batch_size=16,
                    shuffle=True,
                    num_workers=8
                    )


net.load_state_dict(checkpoint['model_state_dict'])
temp=checkpoint["temp"]

net.eval()

Minade = 0
Minfde = 0
Offroad = 0
Offroad_count = 0
val_batch_count = 0


for epoch in range(10):

    with torch.no_grad():
        # Load batch
        for k, data_val in enumerate(test_dl):

            min_ade,min_fde,off_road,off_road_count,count=net(data_val,temp=temp,type=type,device=device,num_samples=1000)

            Minade += min_ade.item()*count
            Minfde += min_fde.item()*count
            Offroad += off_road.item()
            Offroad_count += off_road_count.item()
            val_batch_count += count

    print("Epoch no:", epoch,
        "| temp", format(temp, '0.5f'),
        "| ade", format(Minade / val_batch_count, '0.3f'),
        "| fde", format(Minfde / val_batch_count, '0.3f'),
        "| offroad_rate", format(1-Offroad / Offroad_count, '0.3f'))