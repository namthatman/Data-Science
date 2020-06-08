Music Audio Classification

A CNN model is to classify music audios into genres. The dataset is GTZAN consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format. GTZAN dataset: http://marsyas.info/downloads/datasets.html

The repository consists of one cnn model (melspectrogram approach), one ann model (feature-table approach), backend by django, and a simple database by sqlite.