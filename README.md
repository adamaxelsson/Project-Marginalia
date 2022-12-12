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

# Marginalia prediction
* Download the pre-trained model `faster_r_cnn_weights.pt` from https://drive.google.com/drive/folders/1_snBot1ZguCXwiy475NzU1Hwr6y_cLt6 and place it into `/Project-Marginalia`.
* Create the folder `Project-Marginalia/data/test_images/` and place in them the test images.
* Create the folder `Project-Marginalia/results/`
* To predict and visualize the marginalias, run ```python3 test.py```

# Marginalia Segmentation
**TODO**

# Word labelling with AttentionHTR
* To label the words with AttentionHTR, follow the instructions from https://github.com/dmitrijsk/AttentionHTR

## Acknowledgements
* We would like to thank Ekta Vats for her supervision and the Centre for Digital Humanities Uppsala [(CDHU)](https://www.abm.uu.se/cdhu-eng) for offering the labelled dataset.

## References
[1]: Dmitrijs Kass and Ekta Vats (2022). AttentionHTR: Handwritten Text Recognition Based on Attention Encoder-Decoder Networks. In *CoRR* abs/2201.09390. https://arxiv.org/abs/2201.09390

## Contact

Adam Axelsson (...)

Liang Cheng (...)

Jonas Frankemölle (jonas.frankemolle.9234@student.uu.se)

Ekta Vats (ekta.vats@abm.uu.se)
