import numpy as np
import streamlit as st
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random
from app_utils.generator import Generator
from app_utils.classifier import Classifier

# torch seed to ensure replicability and 'freeze' noise vector
torch.manual_seed(4737)

n_classes = 40 # CelebA number of attributes
z_dim = 64 # noise vector size
batch_size = 128 
device = 'cpu'

# CelebA attributes list
feature_names = ["5oClockShadow", "ArchedEyebrows", "Attractive", "BagsUnderEyes", "Bald", "Bangs",
"BigLips", "BigNose", "BlackHair", "BlondHair", "Blurry", "BrownHair", "BushyEyebrows", "Chubby",
"DoubleChin", "Eyeglasses", "Goatee", "GrayHair", "HeavyMakeup", "HighCheekbones", "Male",
"MouthSlightlyOpen", "Mustache", "NarrowEyes", "NoBeard", "OvalFace", "PaleSkin", "PointyNose", 
"RecedingHairline", "RosyCheeks", "Sideburn", "Smiling", "StraightHair", "WavyHair", "WearingEarrings", 
"WearingHat", "WearingLipstick", "WearingNecklace", "WearingNecktie", "Young"]



def main():

    """ 
    style.css to center the image 
    """

    with open("Static/css/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True) 

    st.title("Controlled Generation Face GAN")

    n_images = 24 # number of image to generate  
    fake_image_history = [] # store generated images
    grad_steps = 20 # number of gradient steps to take
    skip = 4 # number of gradient steps to skip in the visualization


    st.sidebar.title('Features')

    default_control_features = ["5oClockShadow", "Bangs", "BlondHair","BrownHair",
                                "BushyEyebrows", "Eyeglasses", "NoBeard", "Smiling",
                                "PaleSkin", "WearingLipstick", "WearingNecklace"]

    control_features= st.sidebar.radio("Choose your filter ",default_control_features)

    target_indices = feature_names.index(str(control_features))

    gen = load_gen_model()
    classifier = load_classif_model()
    opt = torch.optim.Adam(classifier.parameters(), lr=0.01)

    noise = get_noise(n_images, z_dim).to(device).requires_grad_()
    for i in range(grad_steps):
        opt.zero_grad()
        fake = gen(noise)
        fake_image_history += [fake]
        fake_classes_score = classifier(fake)[:, target_indices].mean()
        fake_classes_score.backward()
        noise.data = calculate_updated_noise(noise, 1 / grad_steps)

    plt.rcParams['figure.figsize'] = [n_images , grad_steps]
    selected_images=[0,1,4,7,9,13,23] # indexes of the number of images to generate (if we have 20 image to generate than the  indexes will be from 0 t 19 if we generate 40 image than it will be from 0 to 39 indexes and we can from 40 images only 10 with indexes between 0 and 39)
    st.image(show_tensor_images(torch.cat(fake_image_history[::skip], dim=2), good_images=selected_images, nrow=n_images).numpy(),width=1000)



@st.cache(allow_output_mutation=True)
def load_gen_model():
    """
    Function to load the pre-trained Generator.
    """
    gen = Generator(z_dim).to(device)
    gen_dict = torch.load("model/dcgan_generator_25.pth", map_location=torch.device(device))["state_dict"]
    gen.load_state_dict(gen_dict)
    gen.eval()
    return gen 

@st.cache(allow_output_mutation=True)
def load_classif_model():
    """
    Function to load the pre-trained Classifier.
    """
    n_classes = 40
    classifier = Classifier(n_classes=n_classes).to(device)
    class_dict = torch.load("model/dcgan_classifier_3_male.pth", map_location=torch.device(device))["state_dict"]
    classifier.load_state_dict(class_dict)
    classifier.eval()    
    return classifier

def calculate_updated_noise(noise, weight):
    '''
    Function to return noise vectors updated with stochastic gradient ascent.
    Parameters:
        noise: the current noise vectors. You have already called the backwards function on the target class
          so you can access the gradient of the output class with respect to the noise by using noise.grad
        weight: the scalar amount by which you should weight the noise gradient
    '''
    new_noise = noise + (noise.grad * weight)
    return new_noise

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples in the batch, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

def show_tensor_images(image_tensor, good_images, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[[good_images]], nrow=nrow)
    image = image_grid.permute(1, 2, 0).squeeze()
    return image


if __name__ == "__main__":
    main()