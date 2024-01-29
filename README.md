# Marginalia-HTR

PyTorch implementation of a Handwritten Text Recognition (HTR) system that focuses on automatic detection and recognition of handwritten marginalia texts i.e., text written in margins or handwritten notes. Faster R-CNN network is used for detection of marginalia and [AttentionHTR](https://github.com/dmitrijsk/AttentionHTR) is used for word recognition. The data comes from early book collections (printed) found in the Uppsala University Library, with handwritten marginalia texts.

For more details, refer to our paper at [arXiv](https://arxiv.org/pdf/2303.05929.pdf).

Liang Cheng, Jonas Frankemölle, Adam Axelsson and Ekta Vats, Uncovering the Handwritten Text in the Margins: End-to-end Handwritten Text Detection and Recognition. To appear in the Proceedings of the 8th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature, co-located with the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2024).

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
* Download the dataset from [here](https://drive.google.com/drive/folders/1_snBot1ZguCXwiy475NzU1Hwr6y_cLt6?usp=share_link).
* Download the pre-trained model `faster_r_cnn_weights.pt` from [here](https://drive.google.com/drive/folders/1k2CxBbIyVp_7iq5-vQgBsP5nOtMSlSIj?usp=sharing) and place it into `/Project-Marginalia/model/`.
* Create the folder `Project-Marginalia/model/results/`
* To detect and visualize the marginalias, run ```python3 model/test.py```

![image](https://github.com/ektavats/Project-Marginalia/assets/73716649/37d7486c-e056-4b85-8680-a9219d56610c)

### Word recognition using AttentionHTR
* To recognise the words with AttentionHTR, follow the instructions from [here](https://github.com/dmitrijsk/AttentionHTR)


## Acknowledgements
* This work was partially supported by the Uppsala-Durham Strategic Development Fund: "Marginalia and Machine Learning: a Study of Durham University and Uppsala University Marginalia Collections".
* The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) partially funded by the Swedish Research Council through grant agreement no. 2022-06725.
* The authors would like to thank Raphaela Heil and Peter Heslin for valuable suggestions and feedback.
* The authors would like to thank Uppsala University Library ([Alvin](https://www.alvin-portal.org/alvin/view.jsf?pid=alvin-organisation%3A16&dswid=-1828)) for offering the dataset and Vasiliki Sampa for the help in preparing the dataset annotation.

## References
[1]: Dmitrijs Kass and Ekta Vats. "AttentionHTR: Handwritten Text Recognition Based on Attention Encoder-Decoder Networks." *International Workshop on Document Analysis Systems*. Springer, Cham, 2022. [Link](https://link.springer.com/chapter/10.1007/978-3-031-06555-2_34) [Code](https://github.com/dmitrijsk/AttentionHTR)

## Contact

Adam Axelsson (adam.axelssons@gmail.com)

Liang Cheng (chengliang653@gmail.com)

Jonas Frankemölle (frankemoelle.jonas@gmail.com)

Ekta Vats (ektavats@gmail.com)
