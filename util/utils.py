import torch
import os


def print_log(result_path, *args):
    os.makedirs(result_path, exist_ok=True)

    print(*args)
    file_path = result_path + '/log.txt'
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)

def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([models.clip.tokenize(text_aug.format(c), context_length=77) for i, c in data.classes.items()])

    return classes

def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images,0.5)
    images = torch.mul(images,2.0)

    return images