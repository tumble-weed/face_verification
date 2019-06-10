# face_verification

- Contains an implementation based on Center Loss and based on the repo https://github.com/KaiyangZhou/pytorch-center-loss
- center loss was chosen because of relative simplicity of implementation over triplet losses etc
- augmentation was done by random brightness, contrast,saturation, and minor hue jitter, further some random rotation, random up and downscales, 
as well as random horizontal flips
- a custom sampler was made to deal with the class imbalance problem

**Testing**

weights must be in a folder called trained_models ( will be saved during training) you'll need to point to a particular epoch
- python test_on_two_images.py image1.png image2.png

**Training**

data must be in a folder called lfw
- python vgg19.py --dataset lfw --gpu 0
