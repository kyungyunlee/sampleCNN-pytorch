## Sample-level Deep CNN
Pytorch implementation of [Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms](https://arxiv.org/abs/1703.01789)

### Data
[MagnaTagATune Dataset](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
* Used tag annotations and audio data

### Model
3^9 model with input sample size 59049  
3 : stride length of the first conv layer (along with filter size 3, it reduces input dimension to 19683)  
9 : 9 hidden conv layers  

### Procedures
* Data processing
    * ` python process_annotations.py `
* Training with cuda
    * ` python main.py --device_num 0 `
	* view loss with ` tensorboard --logdir runs`
* Testing with cuda
	* predict tags for given songs
    * ` python evaluate.py --device_num 0 `


### References
* [https://github.com/jongpillee/sampleCNN](https://github.com/jongpillee/sampleCNN)
* [https://github.com/tae-jun/sample-cnn](https://github.com/tae-jun/sample-cnn)
* [https://github.com/keunwoochoi/magnatagatune-list](https://github.com/keunwoochoi/magnatagatune-list)


