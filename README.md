## Sample-level Deep CNN
Pytorch implementation of [Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms](https://arxiv.org/abs/1703.01789)

### Data
[MagnaTagATune Dataset](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
* Used tag annotations and audio data

### Model
9 1D conv layers and input sample size of 59049 (~3 seconds) 

### Procedures
* Fix `config.py` file
* Data processing
    * run ` python audio_processor.py ` :  audio (to read audio signal from mp3s and save as npy) 
    * run ` python annot_processor.py ` :  annotation (process redundant tags and select top N=50 tags)
		* this will create and save train/valid/test annotation files 
* Training
	* You can set multigpu option by listing all the available devices
    * Ex. ` python main.py --gpus 0 1`
	* Ex. ` python main.py ` will use 1 gpu if available as a default 

### Tag prediction
* run `python eval_tags.py --gpus 0 1 --mp3_file "path/to/mp3file/to/predict.mp3" ` 

### References
* [https://github.com/jongpillee/sampleCNN](https://github.com/jongpillee/sampleCNN)
* [https://github.com/tae-jun/sample-cnn](https://github.com/tae-jun/sample-cnn)
* [https://github.com/keunwoochoi/magnatagatune-list](https://github.com/keunwoochoi/magnatagatune-list)


