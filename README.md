
# Emotion recognition in music using deep neural networks

The accompanying GoogleColab notebook (music_emotion_recognition_cnn.ipynb) contains the main part of the code described in its corresponding paper which investigates the task of emotion recognition in music using deep neural networks with transfer learning techniques and GANs for data augmentation.

## Description

To carry out the classification experiments we used two audio datasets which we converted into Mel-spectrograms (not included here) to use them as inputs to the deep neural networks.
We named those datasets as Big-set and 360-set:

BIG-SET
Big-set includes 17000 audio clips from songs in rock and pop styles, each lasting 10 seconds, from a random part of each song. Entire big-set belongs to a private collection (not publicly available) with their metadata coming from Spotify API.

360-SET
The 360-set excerpts are film soundtracks from 110 films selected from and used by Eerola & Vuoskoski on their study on the discrete and dimensional model of musical emotion classification.

--

The model architectures we used were as follows:
ResNeXt101-32x8d, AlexNet, VGG16bn, SqueezeNet1.0, DenseNet121 and Inception v3, 

EXPERIMENT A
We performed 4 feature classification tasks belonging to the two main types of music emotion classification (discrete and dimensional). The discrete model is a 5-class classification task of Emotions (anger, fear, happy, sad, tender) and the dimensional model is a 3-class classification task of three classes Energy (high, medium, low), Valence (positive, neutral, negative), Tension (high, medium, low). The 6 used architectures (ResNeXt101-32x8d, AlexNet, VGG16bn, SqueezeNet1.0, DenseNet121, Inception v3) and their pre-trained models were taken from the torchvision library. The classifications were performed in both sets with two variations of each model: either by updating the weights of all its layers (whole) or by ”freezing” everything except the classifier layer (freeze). Algorithms were used to find an optimal learning rate and to optimize the initially selected one during training. Max number of epochs was 20 in each training.

EXPERIMENT B
Same scenario as experiment A but this time we pre-trained the models on the big-set instead of ImageNet and performed two classifications on the Energy and Valence features since they are common to both sets.

EXPERIMENT C
Experiment X is about a) creating artificial samples via StyleGAN2-ADA to augment the Emotions feature dataset (because it was the smallest) and b) conducting comparison classification tasks as in experiment A. Again, the dataset produced by the StyleGAN2-ADA or the corresponding models are omitted in this repo.




## Help

You can't actually run the code of the notebook "as is" because all datasets and  CNNs models are omitted. If you want to try the code with another dataset make sure that it meets the Pytorch's torchvision.datasets.ImageFolder class rules.

## Authors

[@ageroul](https://github.com/ageroul)
[@tyiannak](https://tyiannak.github.io/)



## License

This project is licensed under the MIT License 

## Acknowledgments

Inspiration, code snippets, etc.
* [Finetuning torchvision models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
* [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch)
* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
