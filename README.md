# Physiognomy Manifesto

All hail god of physiognomy! Long live physiognomy! May the king of pseudosciences reign for a thousand years! I pray to him when I run out of research ideas. I pray to him when I desperately need a publication. I thank him because he provides me theoretical frameworks to use when I know nothing about social psychology. I am truly grateful for his help in inventing more snake oils! Yes, all kinds of snake oils... from criminality detection, sexual orientation detection, political orientation detection, personality detection and more! All through the cheap and omnipresent facial images on social media... I want to drown these oils. Yes, in his blessings and anointings! Anoint me with more publications! More research grants! More consulting money! More talks! More conference papers! Hallelujah! Amituofo! To the god of physiognomy be the glory and praise forever and ever!

## Installation

The easiest way to enjoy the salvation of physiognomy is to download it from [`PyPI`](https://pypi.org/project/physiognomy/).

```python
pip install physiognomy
```

#### Recommendations for dependency installation

Run using a virtual environment:

```bash
module load python-anaconda3
conda deactivate
conda create -n physiognomy python=3.6 scipy tensorflow=2.3.0 numpy=1.18.5 pandas=1.0.5 opencv-python-headless=4.2.0.34 dlib=19.21.0 imutils=0.5.3 scikit-learn=0.21.3 
source activate physiognomy
```

## Usage

To let the hand of physiognomy move in your research:

```python
from physiognomy.utils import get_rotated_image
from mtcnn import MTCNN
import matplotlib.pyplot as plt
detector = MTCNN()
img = plt.imread("img1.jpg")
bbox = detector.detect_faces(img)[0]['box'] 
plt.imshow(get_rotated_image(img,pt1,pt2,(40,10),(184,10)))
plt.show()
```

Image preprocessing functions in `utils.py`. Some statistic functions in `stats.py`

## Support

Apart from praying to him before you sleep, citing him in your publications, you can also star⭐️ this GitHub repository. Let physiognomy reign forever and ever!