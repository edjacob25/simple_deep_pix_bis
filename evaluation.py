from configparser import ConfigParser

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from nn.deep_pix_bis import DeepPixBiS
from nn.training_data_folder import DataFolderPixBiS


def main():
    config = ConfigParser()
    config.read("config.ini")

    device = "cuda"

    data_folder = config["ROUTES"]["base_data"]
    protocol_folder = f"{config['ROUTES']['base_files']}/Protocols"
    testing_dataset = DataFolderPixBiS(data_folder=data_folder,
                                       groups=['test'],
                                       transform=transforms.ToTensor(),
                                       protocol="Protocol_4",
                                       purposes=['real', 'attack'],
                                       allow_missing_files=True, do_balance=True,
                                       max_samples_per_file=15,
                                       channels='RGB', mask_op='flat', custom_size=14,
                                       protocol_folder=protocol_folder,
                                       partition=1)
    batch_size = 1
    num_workers = 8
    data = DataLoader(testing_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model_path = config["ROUTES"]["model"]
    network = DeepPixBiS(pretrained=True)
    network.load_state_dict(torch.load(model_path)["state_dict"])
    network.to(device)

    with torch.no_grad():
        network.eval()
        accuracy = 0
        for images, labels in data:
            images_v = Variable(images['image'].to(device))

            out = network.forward(images_v)
            map_res = out[0].cpu().numpy()
            res = map_res.flatten().mean()
            print("------------------")
            print(f"{res} - {labels['binary_target']} - {labels['name']}")

        print(accuracy)


if __name__ == '__main__':
    main()
