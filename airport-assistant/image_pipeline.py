# image preprocessing- preparing images to provide them to CLIP in the expected format.

import numpy as np
from PIL import Image
import torchvision.transforms as T

#CLIP mean values for RGB respectively to normalize the CLIP images
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
#CLIP standard deviation values for RGB respectively to normalize the CLIP images
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

#defining a function to load images from the path
def load_image(path):
    # PIL opens the file. convert("RGB") as CLIP expects 3 channels images and it even applies for Black and white images
    image = Image.open(path).convert("RGB")
    return image

#defining a function called resize_image to convert the image size as per the requirement of the CLIP. CLIP was trained on 224*224 image size and even if size=224 is not defined, it will take 224 as default value.
def resize_image(image, size=224):
    return image.resize((size, size), Image.Resampling.BILINEAR)

#defining a function called augment_image to make small changes in the image brightness, contrast and saturations to create variations of the images.
#This helpes the model to be more robust to different versions of images.
def augment_image(image):
    aug = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2)
    return aug(image)

#definind a function to convert images into the tensors format used by PyTorch tensor
def image_to_tensor(image, normalise=True):
    #converting pixels from 0-255 into 0-1 and Height*Width*Channels to Channels*Height*Width
    to_tensor = T.ToTensor()
    tensor = to_tensor(image)
    if normalise:
        tensor = T.Normalize(CLIP_MEAN, CLIP_STD)(tensor)
    return tensor

#defining function to preprocess the images and call previously defined functions
def preprocess_image(path, augment=False):
    image = load_image(path)
    image = resize_image(image)
    if augment:
        image = augment_image(image)
    tensor = image_to_tensor(image, normalise=True)
    return tensor

#defining a function to convert back from tensor to image
def tensor_to_image(tensor):
    # undoing CLIP normalise to disply the results. Tensor cloning is performed not to modify original tensor the tensorso we can plt.imshow the result. Undoing everthing done in previous steps.
    img = tensor.detach().cpu().clone()
    mean = np.array(CLIP_MEAN).reshape(3, 1, 1)
    std = np.array(CLIP_STD).reshape(3, 1, 1)
    arr = img.numpy() * std + mean
    #forcing values to remain between 0 and 1 to avoid invalid image values
    arr = np.clip(arr, 0, 1)
    #converts back from Channels*Height*Width to Height*Width*Channels
    arr = np.transpose(arr, (1, 2, 0))
    #converts backs from 0-1 to 0-255
    arr = (arr * 255).astype(np.uint8)
    #return back from numpy to PIL image
    return Image.fromarray(arr)
