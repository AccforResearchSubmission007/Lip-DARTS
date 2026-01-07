import os
import torch
import argparse
import logging
import torchvision.datasets as dset
from model_search import Network
import utils
import sys


def load_pretrained_supernet(weights_path, init_channels=16, layers=8, num_classes=10):
    criterion = torch.nn.CrossEntropyLoss()
    model = Network(init_channels, num_classes, layers, criterion, fixed_alphas=True)

    if torch.cuda.is_available():
        model = model.cuda()
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')

    model.load_state_dict(checkpoint)
    logging.info("Pretrained supernet weights loaded successfully")

    return model


def fast_search(model, valid_loader, max_batches=15, sa_max_iter=200,
                sa_temp_init=10.0, sa_cooling_rate=0.95):
    logging.info("Starting fast architecture search...")

    model.eval()
    model.update_path_strength_valid(
        valid_loader,
        max_batches=max_batches,
        sa_max_iter=sa_max_iter,
        sa_temp_init=sa_temp_init,
        sa_cooling_rate=sa_cooling_rate
    )

    genotype = model.genotype()
    logging.info(f"Searched genotype: {genotype}")

    return genotype


def main():
    parser = argparse.ArgumentParser(description="Fast Architecture Search")
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained weights')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--max_batches', type=int, default=5, help='maximum number of batches')
    parser.add_argument('--sa_max_iter', type=int, default=1200, help='SA max iterations')
    parser.add_argument('--sa_temp_init', type=float, default=10.0, help='SA initial temperature')
    parser.add_argument('--sa_cooling_rate', type=float, default=0.95, help='SA cooling rate')
    parser.add_argument('--save_dir', type=str, default='fast_search_results', help='experiment name')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout data augmentation')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'fast_search_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not os.path.exists(args.weights):
        logging.error(f"Weights file not found: {args.weights}")
        return

    model = load_pretrained_supernet(
        args.weights,
        init_channels=args.init_channels,
        layers=args.layers
    )

    logging.info("Preparing validation data...")
    _, valid_transform = utils._data_transforms_cifar10(args)
    valid_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=valid_transform)

    num_valid = len(valid_data)
    indices = list(range(num_valid))
    split = int(0.5 * num_valid)

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2
    )

    genotype = fast_search(
        model,
        valid_loader,
        max_batches=args.max_batches,
        sa_max_iter=args.sa_max_iter,
        sa_temp_init=args.sa_temp_init,
        sa_cooling_rate=args.sa_cooling_rate
    )

    result_file = os.path.join(args.save_dir, 'searched_genotype.txt')
    with open(result_file, 'w') as f:
        f.write(str(genotype))

    logging.info(f"Genotype saved to: {result_file}")
    logging.info("Fast architecture search completed!")


if __name__ == '__main__':
    main()