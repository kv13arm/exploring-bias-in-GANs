# Exploring bias in GANs

## Project overview
Streamlit app to oncover feature entanglement in the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and confirm inherent bias.

## Generative Adversarial Networks
Generative Adversarial Networks or GANs are machine learning architectures where two deep neaural networks compete with each other to create realistic outputs. GANs achieve this level of realism by pairing a generator, which learns to produce the target output, with a discriminator, which learns to distinguish true data from the output of the generator. The generator tries to fool the discriminator, and the discriminator tries to keep from being fooled.

For this project, the GAN type used to generate human-like faces is a Deep Convolutional Generative Adversarial Network (DCGAN). This type of GAN includes convolutional layers in its architecture and is more adapted for image datasets.

Here are some of the outputs of the model:
![Generated Images](https://raw.githubusercontent.com/k13var/exploring-bias-in-GANs/main/img/generated_img.png)

Aside from producing agnostic content (e.g. human faces), GANs can equally be leveraged to produce controllable outputs (e.g. human faces with specific traits like smiling), see the Streamlit app. 

## Bias in machine learning
By controlling with a classifier the amount of a desired (target) feature to output one can observe how this will influence the amount of another, non-target, feature that is outputed as a by-product. This approach aims to detect potential bias by analysing correlations in feature classifications on the generated images.

*Note that not all biases are as obvious as the ones presented below.*

When controlling the amount of 'female-ness'[^bignote], the figure below shows how much the other features change, as detected in those images by the classifier. 

![Feature correlation graph](https://raw.githubusercontent.com/k13var/exploring-bias-in-GANs/main/img/female-ness_bias.png)

Therefore, more of the 'Female' feature will correspond, for example, to more of the 'HeavyMakeup' feature but less of the 'Bald' and 'WearingNecktie' features. 

The following table ranks the top 10 features that covary with the 'Female' feature the most. This includes large negative and positive covariances.

|     Feature     | Covarience wt 'Female' |
|:----------------|-----------------------:|
| WearingLipstick |               2.839798 |
| HeavyMakeup     |               2.501494 |
| NoBeard         |               2.127061 |
| 5oClockShadow   |              -1.849787 |
| WearingNecktie  |              -1.733796 |
| Sideburn        |              -1.730549 |
| Goatee          |               1.519677 |
| Mustache        |              -1.502830 |
| Bald            |              -1.296670 |
| BlondHair       |               1.246212 |

## How to run the Streamlit application
The app allows to visualise the bias in the CelebA dataset and gain awareness of the possibility of introducing bias in machine learning model if it left unchecked. 

For example, by selecting to add more of a feature, the generated face can turn from masculine to feminine or the skin colour can vary from lighter to darker. 

To run the app, follow the instructions below:
```
git clone https://github.com/k13var/exploring-bias-in-GANs.git
cd exploring-bias-in-GANs
pip install -r requirements.txt
streamlit run app.py
```

## Tools, modules and techniques
**Python - web development:**

Streamlit

**Python - machine learning**

pytorch | numpy | matplotlib 

#### Ressources
[DeepLearning.AI Generative Adversarial Networks (GANs) Specialization](https://www.deeplearning.ai/generative-adversarial-networks-specialization/)
[DCGAN TUTORIAL!](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)


[^bignote]: The CelebA does not have a 'Female' feature but it has a 'Male' one. By reversing the label values in the 'Male' feature we were able to control feature selection based on a 'Female' feature thus created.