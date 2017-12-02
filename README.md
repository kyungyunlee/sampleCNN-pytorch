## Sample-level Deep CNN
Pytorch implementation of [Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms](https://arxiv.org/abs/1703.01789)

### Data
[MagnaTagATune Dataset](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)

* Used tag annotations and audio data

### Model
3^9 model with input sample size 59049

3 : stride length of the first conv layer (along with filter size 3, it reduces input dimension to 19683)

9 : 9 hidden conv layers

### Steps
* Data processing
```
    python process_audio.py
    python process_annotations.py
```
* Training/Validating
```
    python main.py
```
* Testing
```
    python evaluate.py
```

### Credits for code references (many many thanks to them!)
* [https://github.com/jongpillee/sampleCNN](https://github.com/jongpillee/sampleCNN)
* [https://github.com/tae-jun/sample-cnn](https://github.com/tae-jun/sample-cnn)
* [https://github.com/keunwoochoi/magnatagatune-list](https://github.com/keunwoochoi/magnatagatune-list)


