import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForImageClassification, AutoConfig
from PIL import Image
from timm.data.transforms_factory import create_transform
import os


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


def get_dir(args):
    output_path = args.output_path
    name = args.name

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    path = os.path.join(output_path, name)
    saved_model_folder = os.path.join(path, "models")

    os.makedirs(saved_model_folder, exist_ok=True)

    return path, saved_model_folder


def setup_ddp():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    dist.destroy_process_group()


def create_dataloader(image_folder, batch_size, im_size, config, is_training=False):
    transform = create_transform(input_size=(3, im_size, im_size), is_training=is_training,
                                 mean=config.mean, std=config.std,
                                 crop_mode=config.crop_mode, crop_pct=config.crop_pct)

    dataset = ImageFolder(root=image_folder, transform=transform)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    return dataloader


def main(args):
    policy = 'color,translation,cutout'
    output_path, root_models = get_dir(args)
    root_train = args.root_train
    root_test = args.root_test
    model_name = args.model_name
    batch_size = args.batch_size
    im_size = args.im_size
    num_classes = args.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setup_ddp()

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForImageClassification.from_pretrained(model_name, config=config, trust_remote_code=True)

    model.to(device)

    model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    train_loader = create_dataloader(root_train, batch_size, im_size, config, is_training=True)
    test_loader = create_dataloader(root_test, batch_size, im_size, config, is_training=False)

    # Определение оптимизатора и функции потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def train_one_epoch(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loader.sampler.set_epoch(epoch)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (images, target) in loop:
            images = images.to(device)
            images = DiffAugment(images, policy)
            target = target.to(device)

            outputs = model(images)['logits']
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (target == target).sum().item()
            total += images.size(0)

            accuracy = correct / total

            loop.set_description(f"Epoch [{epoch}/{args.num_epochs}]")
            loop.set_postfix(accuracy = accuracy)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total

        loop.set_postfix(epoch_loss = epoch_loss, epoch_accuracy = epoch_accuracy)

        save_checkpoint(model.module, optimizer, root_models + "/model.pth.tar")

        if int(os.environ['LOCAL_RANK']) == 0:
            model.module.save_pretrained(root_models + "/mamba_tiny2_1k.pth.tar")

    
    def evaluate(epoch):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            test_loader.sampler.set_epoch(epoch)
            loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
            for bathc_idx, (images, target) in loop:
                images = images.to(device)
                target = target.to(device)

                outputs = model(images)['logits']
                
                correct += (outputs == target).sum().item()
                total += images.size(0)
        
        accuracy = correct / total
        loop.set_postfix(test_accuracy = accuracy)

    
    num_epochs = args.num_epochs

    for epoch in range(num_epochs):
        train_one_epoch(epoch)
        evaluate(epoch)

    cleanup_ddp()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='mambavision')

    parser.add_argument('--root_train', type=str, default="/kaggle/working/split_data/train")
    parser.add_argument('--root_test', type=str, default="/kaggle/working/split_data/test")
    parser.add_argument("--output_path", type=str, default='./')
    parser.add_argument('--name', type=str, default='results', help='experiment name')
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--model_name", type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--im_size', type=int, default=224)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()
    main(args)