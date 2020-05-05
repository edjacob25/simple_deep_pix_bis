from configparser import ConfigParser
from pathlib import Path
from shutil import rmtree

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from nn.deep_pix_bis import DeepPixBiS
from nn.generic_trainer import GenericTrainer
from nn.training_data_folder import DataFolderPixBiS


def compute_loss(network, img, labels, device):
    """
    Compute the losses, given the network, data and labels and
    device in which the computation will be performed.
    """
    criterion_pixel = nn.BCELoss()

    criterion_bce = nn.BCELoss()

    images_v = Variable(img['image'].to(device))
    # print(images_v.shape)
    labels_v_pixel = Variable(labels['pixel_mask'].to(device))

    labels_v_binary = Variable(labels['binary_target'].to(device))

    out = network(images_v)

    beta = 0.5

    loss_pixel = criterion_pixel(out[0].squeeze(1), labels_v_pixel.float())

    loss_bce = criterion_bce(out[1], labels_v_binary.unsqueeze(1).float())

    loss = beta * loss_bce + (1.0 - beta) * loss_pixel

    return loss


def main():
    config = ConfigParser()
    config.read("config.ini")

    img_transform = {'train': transforms.Compose(
        [transforms.ToPILImage(), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0),
         transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        'val': transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}

    for partition in range(1, 7):
        data_folder = config["ROUTES"]["base_data"]
        protocol_folder = f"{config['ROUTES']['base_files']}/Protocols"
        training_dataset = DataFolderPixBiS(data_folder=data_folder,
                                            transform=img_transform["train"],
                                            groups=['train'],
                                            protocol="Protocol_4",
                                            purposes=['real', 'attack'],
                                            allow_missing_files=True, do_balance=True,
                                            max_samples_per_file=15,
                                            channels='RGB', mask_op='flat', custom_size=14,
                                            protocol_folder=protocol_folder,
                                            partition=partition)

        validation_dataset = DataFolderPixBiS(data_folder=data_folder,
                                              transform=img_transform["train"],
                                              groups=['dev'],
                                              protocol="Protocol_4",
                                              purposes=['real', 'attack'],
                                              allow_missing_files=True, do_balance=True,
                                              max_samples_per_file=8,
                                              channels='RGB', mask_op='flat', custom_size=14,
                                              protocol_folder=protocol_folder,
                                              partition=partition)

        batch_size = 32
        num_workers = 8
        epochs = 5
        learning_rate = 0.0001
        weight_decay = 0.000001
        save_interval = 2

        network = DeepPixBiS(pretrained=True)
        network.cuda()
        for name, param in network.named_parameters():
            param.requires_grad = True

        # optimizer initialization

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=learning_rate,
                               weight_decay=weight_decay)

        data = {"train": DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True),
                "val": DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)}

        trainer = GenericTrainer(network, optimizer, compute_loss, learning_rate, batch_size, "cuda", True,
                                 save_interval=save_interval)

        output_dir = Path(f"out/Partition{partition}")

        if output_dir.exists():
            rmtree(output_dir)
        output_dir.mkdir()

        trainer.train(data, epochs, output_dir=output_dir)


if __name__ == '__main__':
    main()
