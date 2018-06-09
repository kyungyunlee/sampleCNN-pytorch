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
    * audio (to read audio signal from mp3s and save as npy) : ` python process_audio.py `
    * annotation (process redundant tags and select top N=50 tags): ` python process_annotations.py `
		* this will create and save train/valid/test annotation files 
* Training
    * ` python main.py --device_num 0 `
* Testing 
	* predict tags for given songs
    * ` python evaluate.py --device_num 0 `

### Tag prediction
* `python eval_tags.py --device_num 0 --mp3_file "path/to/mp3file/to/predict.mp3" ` 

### References
* [https://github.com/jongpillee/sampleCNN](https://github.com/jongpillee/sampleCNN)
* [https://github.com/tae-jun/sample-cnn](https://github.com/tae-jun/sample-cnn)
* [https://github.com/keunwoochoi/magnatagatune-list](https://github.com/keunwoochoi/magnatagatune-list)


