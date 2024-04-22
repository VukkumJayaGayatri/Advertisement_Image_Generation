import tensorflow as tf
import pickle

# Path to the downloaded pretrained model file (.pkl)
model_path = '~/stylegan2_models/stylegan2-ffhq-config-f.pkl'

# Load the network from the pretrained model file
with open(model_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].clone().requires_grad_(False).to('cuda')

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = G.to(device)

import numpy as np

# Number of images to generate
num_images = 5

# Generate random latent vectors
latent_vectors = np.random.randn(num_images, G.input_shape[1])

# Generate images using the generator network
images = Gs.run(latent_vectors, None, truncation_psi=0.7, randomize_noise=True, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))

from PIL import Image

# Save generated images to disk
for i, img in enumerate(images):
    im = Image.fromarray(img)
    im.save(f'generated_image_{i}.png')
