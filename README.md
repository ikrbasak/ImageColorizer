# Image Colorizer
## Introduction
Pix2Pix is a strong model for image-to-image translation which can be optimized for colorization. One improvement is to use Wasserstein GAN instead of a traditional GAN, as WGANs utilize the Wasserstein distance metric for training the generator and discriminator, resulting in a more stabilized training process and more realistic outputs. Another approach to enhancing Pix2Pix for colorization is to use a U-Net architecture based on residual blocks. U-Net is a CNN optimized for image segmentation and consists of convolutional and max pooling layers with skip connections between layers of equal resolution, allowing for fine details of the input image to be learned. Residual blocks are layers in neural networks with skip connections that permit easier gradient passage, leading to faster convergence and better results. Integrating WGAN with a U-Net architecture based on residual blocks can improve Pix2Pix's performance in colorization by providing improved stability and detail-learning capability.

## Image Colorization
Image colorization refers to the task of adding color to a black and white or grayscale image. This is achieved by converting the grayscale intensity values to a color format, such as RGB, and then completing the color channels to create a full-color image. There are several methods to colorize images, many of which involve image processing techniques like segmentation, texture synthesis, or machine learning. A widely used method is to utilize a deep learning approach, specifically a convolutional neural network (CNN), to colorize images. This involves training a CNN on a large set of color images, then using the network to fill in the missing color channels of a grayscale image. Another method is through the use of a Generative Adversarial Network (GAN) model where a generator network creates the color version of the grayscale image and a discriminator network is trained to differentiate between the generated image and a real color image. In recent years, deep learning methods for image colorization have produced impressive results and can generate high-quality colorization results on various types of images.

## GAN
A generative adversarial network (GAN) is a type of Deep Learning (DL) network that can generate data with similar characteristics as the input training data.

A GAN consists of two networks that train together:

1. **Generator —** Given a vector of random values as input, this network generates data with the same structure as the training data.
2. **Discriminator —** Given batches of data containing observations from both the training data, and generated data from the generator, this network attempts to classify the observations as "real" or "generated".

![Simple GAN Architecture](https://it.mathworks.com/help/examples/nnet/win64/TrainConditionalGenerativeAdversarialNetworkCGANExample_01.png "Simple GAN Architecture")

## Conditional GAN
A conditional generative adversarial network (CGAN) is a type of GAN that also takes advantage of labels during the training process.

1. **Generator —** Given a label and random array as input, this network generates data with the same structure as the training data observations corresponding to the same label.
2. **Discriminator —** Given batches of labeled data containing observations from both the training data and generated data from the generator, this network attempts to classify the observations as "real" or "generated".

![CGAN Architecture](https://it.mathworks.com/help/examples/nnet/win64/TrainConditionalGenerativeAdversarialNetworkCGANExample_02.png "CGAN Architecture")

To train a conditional GAN, train both networks simultaneously to maximize the performance of both:

- Train the generator to generate data that "fools" the discriminator.
- Train the discriminator to distinguish between real and generated data.

To maximize the performance of the generator, maximize the loss of the discriminator when given generated labeled data. That is, the objective of the generator is to generate labeled data that the discriminator classifies as "real".

To maximize the performance of the discriminator, minimize the loss of the discriminator when given batches of both real and generated labeled data. That is, the objective of the discriminator is to not be "fooled" by the generator.

Ideally, these strategies result in a generator that generates convincingly realistic data that corresponds to the input labels and a discriminator that has learned strong feature representations that are characteristic of the training data for each label.

[View source](https://it.mathworks.com/help/deeplearning/ug/train-conditional-generative-adversarial-network.html)

## GAN vs CGAN
Conditional Generative Adversarial Networks (CGANs) are a variation of regular Generative Adversarial Networks (GANs) that are designed to deal with conditional data. Just like a standard GAN, a CGAN has a generator and discriminator network. However, the generator and discriminator in a CGAN are both conditioned on some supplementary input data. This additional data can be utilized to regulate the generator's output, allowing it to produce more tailored or specific results.

CGANs offer several advantages over standard GANs:
1. **Control on generated data:** The generator's output in a CGAN is influenced by the input data, providing more control and specificity in the model's output. For instance, if the input is a grayscale image, the model can color it according to a specific color palette.
2. **Improved stability and training:** With the generator being conditioned on extra input data, it can be more stable and easier to train than a standard GAN. This is because the generator concentrates on a particular subset of the data instead of trying to generate all possible outputs.
3. **Handling missing data:** CGANs are well-suited for handling missing data or data with missing modalities. The extra input data can be used to condition the generator to produce plausible outputs for the missing data.
4. **Handling multiple classes:** CGANs can generate data for multiple classes in a one-to-many mapping where the generator is conditioned on the class label and produces an image from that class.
5. **Handling conditional data:** In some tasks, the data is conditional, such as in image-to-image translation, where the output depends on the input. CGANs are capable of dealing with this type of conditional data effectively.

## Generator
### U-Net
The U-Net is a neural network architecture for image segmentation that prioritizes speed and accuracy. It's still widely used in the field of semantic segmentation today and has been successful in numerous challenges.

The U-Net consists of two parts: an encoder path (the backbone) and a decoder path. 

- The encoder captures image features at different scales by using stacks of convolutional and max pooling layers. It repeats two convolutional layers followed by a non-linearity layer and a max pooling layer, doubling the number of feature maps in each block to capture complex structures.
- The decoder path expands symmetrically and uses transposed convolutions. This type of convolutional layer is an up-sampling method that's the reverse of max pooling. It reduces the number of feature maps in each block and each convolution block is followed by an up-convolutional layer. The feature maps from the corresponding encoder block are appended to the output of each up-convolutional layer to make it easier for the network to recreate the segmentation mask. If the dimensions of the encoder feature maps exceed the decoder ones, they are cropped. 
 
[Source](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862)

### UNet with ResBlock
The UNet architecture was a significant advancement in computer vision that transformed the field of image segmentation, especially in medical imaging, but also in other areas. Its defining characteristic is the extended connections between the contracting and expanding paths, which functions like FCN that is lifted from both ends.

ResNet, on the other hand, was another major breakthrough in computer vision, characterized by its residual blocks and skip connections. This allowed the creation of deeper convolutional neural networks, leading to record-breaking results in image classification on the ImageNet dataset.

![ResNet](https://miro.medium.com/max/720/0*Q6Dq_Ztsno3zV8TF "ResNet")

By incorporating ResBlocks in place of the convolutions at each level in U-Net, the performance of the model is typically improved compared to the original U-Net. The detailed model architecture is shown in the diagram.

![UNet with ResBlock](https://i.imgur.com/k6ErEni.png "UNet with ResBlock")

[Source](https://medium.com/@nishanksingla/unet-with-resblock-for-semantic-segmentation-dd1766b4ff66)

## Discriminator
The discriminator or critic network uses a conventional CNN design, where the input image is processed through a series of convolutions, batch normalization, LeakyReLU activation, and downsampling. The convolutions extract features from the image, while the downsampling reduces the spatial dimensions of the features and increases the number of filters.

The critic network operates on images in the LAB color space, which is the concatenation of the ab channels and the L channel. The output of the network is a scalar indicating whether the input image is real or fake.

The architecture of the critic network is crucial for image recoloring as it enables the generator network to learn the distribution of real images in the LAB color space. The critic network provides an accurate evaluation of the generated images, helping the generator produce more realistic and higher-quality outputs. The use of LeakyReLU activation and Instance Normalization also enhances the performance of the critic network by stabilizing the training and reducing mode collapse.

![Discriminator](https://i.imgur.com/rG6DjQA.png "Discriminator")

## Others
1. Inception Score (IS) - [(1)](https://medium.com/octavian-ai/a-simple-explanation-of-the-inception-score-372dff6a8c7a) [(2)](https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/)
2. Frechet Inception Distance (FID) - [(1)](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/) [(2)](https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI) [(3)](https://github.com/mseitzer/pytorch-fid) 

## Try the Colorizer
Use the image colorizer [here](https://imgcolor.pythonanywhere.com/).
<details><summary>Note</summary>
<p>
The colorizer is hosted on <a href="https://www.pythonanywhere.com/">https://www.pythonanywhere.com/ (PWA)</a> for taking inference. PWA is not meant for hosting ML/DL models and therefore sometimes it shows <i>503 - Service Unavailable</i> error. I will try to host using some proper hosting service.
</p>
</details>

## Todo:
- [ ] Utilize more data and other dataset
- [ ] Tune hyper-parameters
- [ ] Change to other hosting
- [ ] Prettify UI
