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
                                       protocol_folder=protocol_folder)
    batch_size = 32
    num_workers = 8
    data = DataLoader(testing_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model_path = config["ROUTES"]["model"]
    network = DeepPixBiS(pretrained=True)
    network.load_state_dict(torch.load(model_path)["state_dict"])
    network.to(device)

    with torch.no_grad():
        network.eval()
        test_loss = 0
        accuracy = 0
        for images, labels in data:
            images_v = Variable(images['image'].to(device))

            labels_v_pixel = Variable(labels['pixel_mask'].to(device))

            labels_v_binary = Variable(labels['binary_target'].to(device))
            # print(images_v.shape)
            out = network.forward(images_v)

            # print("B")

            criterion_pixel = nn.BCELoss()

            criterion_bce = nn.BCELoss()

            loss_pixel = criterion_pixel(out[0].squeeze(1), labels_v_pixel.float())

            loss_bce = criterion_bce(out[1], labels_v_binary.unsqueeze(1).float())

            beta = 0.5

            test_loss += beta * loss_bce + (1.0 - beta) * loss_pixel

            # print(out[1])
            # print(labels['binary_target'])

            ps = torch.exp(out[1])
            # print(labels_v_binary.unsqueeze(1).float())
            # print(ps)
            equality = (labels_v_binary.unsqueeze(1).float() == ps.max(dim=1)[1])
            # print(equality)
            accuracy += equality.mean()
        print(accuracy)


if __name__ == '__main__':
    main()
