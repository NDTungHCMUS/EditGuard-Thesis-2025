import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torchvision.transforms as transforms
from .feature_extractors import ResNet18Embedding, VAEEmbedding, ClipEmbedding, KLVAEEmbedding
import argparse


EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200
BATCH_SIZE = 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        choices=["resnet18", "clip", "klvae8", "sdxlvae", "klvae16"],
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=2,
        choices=[2, 4, 6, 8],
    )
    parsed_args = parser.parse_args()
    return parsed_args


def adv_emb_attack(
    wm_img_path, encoder, strength, output_path, device=torch.device("cuda:0")
):
    # check if the file/directory paths exist
    for path in [wm_img_path, output_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path does not exist: {path}")

    # load embedding model
    if encoder == "resnet18":
        # we use last layer's state as the embedding
        embedding_model = ResNet18Embedding("last")
    elif encoder == "clip":
        embedding_model = ClipEmbedding()
    elif encoder == "klvae8":
        # same vae as used in generator
        embedding_model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
    elif encoder == "sdxlvae":
        embedding_model = VAEEmbedding("stabilityai/sdxl-vae")
    elif encoder == "klvae16":
        embedding_model = KLVAEEmbedding("kl-f16")
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")
    embedding_model = embedding_model.to(device)
    embedding_model.eval()
    print("Embedding Model loaded!")

    # load data
    transform = transforms.ToTensor()
    wm_dataset = SimpleImageFolder(wm_img_path, transform=transform)
    wm_loader = DataLoader(
        wm_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    print("Data loaded!")

    # Create an instance of the attack
    attack = WarmupPGDEmbedding(
        model=embedding_model,
        eps=EPS_FACTOR * strength,
        alpha=ALPHA_FACTOR * EPS_FACTOR * strength,
        steps=N_STEPS,
        device=device,
    )

    # Generate adversarial images
    for i, (images, image_paths) in enumerate(wm_loader):
        images = images.to(device)

        # PGD attack
        images_adv = attack.forward(images)

        # save images
        for img_adv, image_path in zip(images_adv, image_paths):
            save_path = os.path.join(output_path, os.path.basename(image_path))
            save_image(img_adv, save_path)
    print("Attack finished!")
    return

def adv_emb_attack_2(
    images: torch.Tensor,
    encoder: str,
    strength: float,
    device: torch.device = torch.device("cuda:0"),
    eps_factor: float = 1/255,
    alpha_factor: float = 0.05,
    n_steps: int = 200,
) -> torch.Tensor:
    # Move to device but don’t suppress grads here
    images = images.to(device)

    # Load & eval the embedding model
    if encoder == "resnet18":
        model = ResNet18Embedding("last")
    elif encoder == "clip":
        model = ClipEmbedding()
    elif encoder == "klvae8":
        model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
    elif encoder == "sdxlvae":
        model = VAEEmbedding("stabilityai/sdxl-vae")
    elif encoder == "klvae16":
        model = KLVAEEmbedding("kl-f16")
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")
    model = model.to(device).eval()

    # Create the attacker
    attacker = WarmupPGDEmbedding(
        model=model,
        device=device,
        eps=eps_factor * strength,
        alpha=alpha_factor * eps_factor * strength,
        steps=n_steps,
    )

    # **No torch.no_grad() here!** We need autograd.
    adv_images = attacker.forward(images)
    return adv_images

class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None, extensions=None):
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]
        self.root = root
        self.transform = transform
        self.extensions = extensions

        # Load filenames from the root
        self.filenames = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and os.path.splitext(f)[1].lower() in self.extensions
        ]

    def __getitem__(self, index):
        image_path = self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path  # return image path to identify the image file later

    def __len__(self):
        return len(self.filenames)


class WarmupPGDEmbedding:
    def __init__(
        self,
        model,
        device,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        loss_type="l2",
        random_start=True,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_type = loss_type
        self.random_start = random_start
        self.device = device

        # Initialize the loss function
        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    # def forward(self, images, init_delta=None):
    #     # 1) send to device
    #     images = images.to(self.device)

    #     # 2) get original embeddings once (no grad needed)
    #     with torch.no_grad():
    #         original_embeddings = self.model(images).detach()

    #     # 3) initialize adv_images in the ε-ball
    #     if self.random_start:
    #         adv_images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
    #         adv_images = adv_images.clamp(0.0, 1.0)
    #     else:
    #         adv_images = images.clone()

    #     # 4) PGD loop
    #     for _ in range(self.steps):
    #         # a) make a fresh leaf that requires gradient
    #         adv_images = adv_images.detach().requires_grad_(True)

    #         # b) forward + compute distance loss in embedding space
    #         adv_embeddings = self.model(adv_images)
    #         loss = self.loss_fn(adv_embeddings, original_embeddings)

    #         # c) compute gradient w.r.t. adv_images
    #         grad = torch.autograd.grad(loss, adv_images)[0]

    #         # d) gradient step & projection (no grad tracking)
    #         with torch.no_grad():
    #             adv_images = adv_images + self.alpha * grad.sign()
    #             delta = (adv_images - images).clamp(-self.eps, self.eps)
    #             adv_images = (images + delta).clamp(0.0, 1.0)

    #     # 5) return clean tensor
    #     return adv_images.detach()

    # def forward(self, images, init_delta=None):
    #     print("Vao trong forward roi")
    #     images = images.to(self.device)
    #     with torch.no_grad():
    #         original_embeddings = self.model(images).detach()

    #     if self.random_start:
    #         adv_images = (images +
    #                         torch.empty_like(images).uniform_(-self.eps, self.eps)
    #                         ).clamp(0,1)
    #     else:
    #         adv_images = images.clone()

    #     for _ in range(self.steps):
    #         adv_images = adv_images.detach().requires_grad_(True)

    #         # now uses a forward WITHOUT any no_grad inside!
    #         adv_embeddings = self.model(adv_images)
    #         loss = self.loss_fn(adv_embeddings, original_embeddings)

    #         grad = torch.autograd.grad(loss, adv_images)[0]

    #         with torch.no_grad():
    #             adv_images = (adv_images + self.alpha * grad.sign())
    #             delta = (adv_images - images).clamp(-self.eps, self.eps)
    #             adv_images = (images + delta).clamp(0,1)

    #     return adv_images.detach()

    def forward(self, images, init_delta=None):
        self.model.eval()
        images = images.clone().detach().to(self.device)

        # Get the original embeddings
        original_embeddings = self.model(images).detach()
        print("Images", images)
        # initialize adv images
        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        # PGD
        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)

            # Calculate loss
            cost = self.loss_fn(adv_embeddings, original_embeddings)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        print("Adv images", adv_images)
        return adv_images

if __name__ == "__main__":
    adv_emb_attack("/workspace/WAVES/input", "klvae8", 2, "/workspace/WAVES/output")