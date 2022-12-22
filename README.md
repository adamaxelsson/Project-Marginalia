# Project-Marginalia

This is a PyTorch implementation to find marginalias, segment words, and label them using the AttentionHTR [1] network. The network to find marginalias is based on a Faster R-CNN network. The network was trained on scanned book pages from Uppsala library.  

## Dependencies 

To run the code, run the following

```
python3 -m venv marginalia-env
source marginalia-env/bin/activate
pip install --upgrade pip
python3 -m pip install -r Project-Marginalia/requirements.txt
```


## Demo of our pre-trained model

### Marginalia prediction
* Download the pre-trained model `faster_r_cnn_weights.pt` from https://drive.google.com/drive/folders/1_snBot1ZguCXwiy475NzU1Hwr6y_cLt6 and place it into `/Project-Marginalia/model/`.
* Create the folder `Project-Marginalia/model/data/test_images/` and place in them the test images.
* Create the folder `Project-Marginalia/model/results/`
* To predict and visualize the marginalias, run ```python3 model/test.py```

### Marginalia Segmentation
* If you want to use your model on your own set of images, use the image_to_bboxes.py script. In it you will have to add the path to your model, folder of your dataset, and the location where you want the predicted marginalia to be saved. Then run 'python3 image_to_bboxes.py'.
* If you want to segment a set of marginalia to individual words, use the marginalia_to_words.py script. In it you will have to add the path to a folder containing images of predicted marginalia, as well as the folder where you want the results to be saved. Then run 'python3 marginalia_to_words.py'.

### Word recognition using AttentionHTR
* To recognise the words with AttentionHTR, follow the instructions from https://github.com/dmitrijsk/AttentionHTR

## Acknowledgements
* This work has been partially supported by the Matariki Network Initiation Grant: "Marginalia and Machine Learning: a Study of Durham University and Uppsala University Marginalia Collections".
* The computations/data handling were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE) partially funded by the Swedish Research Council through grant agreement no. 2018-05973, project Dnr: SNIC 2022/22-1084.
* The authors would like to thank [Ekta Vats](https://www.ektavats.se/) for her supervision, and the Centre for Digital Humanities Uppsala ([CDHU](https://www.abm.uu.se/cdhu-eng)) and Uppsala University Library ([Alvin](https://www.alvin-portal.org/alvin/view.jsf?pid=alvin-organisation%3A16&dswid=-1828)) for offering the dataset.


## References
[1]: Dmitrijs Kass and Ekta Vats. "AttentionHTR: Handwritten Text Recognition Based on Attention Encoder-Decoder Networks." *International Workshop on Document Analysis Systems*. Springer, Cham, 2022. [Link](https://link.springer.com/chapter/10.1007/978-3-031-06555-2_34) [Code](https://github.com/dmitrijsk/AttentionHTR)

## Contact

Adam Axelsson (adam.axelsson.4529@student.uu.se)

Liang Cheng (liang.cheng.8263@student.uu.se)

Jonas Frankemölle (jonas.frankemolle.9234@student.uu.se)

Ekta Vats (ekta.vats@abm.uu.se)
