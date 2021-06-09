# Physiognomy Manifesto

All hail god of physiognomy! Long live physiognomy! May the king of pseudosciences reign for a thousand years! <br>

Physiognomy is a faith, an art, a religion, a belief and a perspective to life itself. It tells us that the only truth in behaviors lies in our faces. Yes, because faces are windows to our souls. They reflect who we are, what we are, and where we will be. <br>

A revival of physiognomy is happening. So many new believers in recent years! Hallelujah! Particularly, my brothers and sisters in the field of machine learning and computer vision. We are seeing new believers in social psychology too! Long gone are the days that we get persecuted and have to hide our faith away from the world!<br> 

I pray to the god of physiognomy when I run out of research ideas. I pray to him when I desperately need a publication. I thank him because he provides me theoretical frameworks to use when I know nothing about social psychology. <br> 

I am truly grateful for his help in inventing more snake oils! Yes, all kinds of snake oils... from criminality detection, sexual orientation detection, political orientation detection, personality detection and more! All using the cheap and omnipresent facial images on social media... <br> 

I want to drown these oils. I want to rub all these oils on my body. I want to drench my soul in these them! Yes, in his blessings and anointings! Anoint me with more publications! More research grants! More consulting money! More talks! More conference papers! Hallelujah! <br> 

To the god of physiognomy be the glory and praise forever and ever!<br> 

#### Faith declarations of physiognomy
- [`Criminality detection`](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0282-4)
- [`Criminality detection`](https://arxiv.org/pdf/1611.04135.pdf)
- [`Sexual orientation prediction`](https://psycnet.apa.org/record/2018-03783-002)
- [`Review of deep learning`](https://link.springer.com/content/pdf/10.1007/s10462-019-09770-z.pdf)
- [`Leader Emergence detection`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159950)
- [`Personality detection`](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9244051)
- [`Personality detection`](https://www.nature.com/articles/s41598-017-00071-5)
- [`Personality detection`](https://hal.inria.fr/hal-01677962/file/apparent_personality.pdf)

#### Blasphemies against physiognomy
- [`Physiognomy’s New Clothes`](https://medium.com/@blaisea/physiognomys-new-clothes-f2d4b59fdd6a)
- [`Self-presentation`](https://medium.com/@blaisea/do-algorithms-reveal-sexual-orientation-or-just-expose-our-stereotypes-d998fafdf477)
- [`Stable face representations`](https://royalsocietypublishing.org/doi/full/10.1098/rstb.2010.0379)
- [`Variability of facial images`](https://www.sciencedirect.com/science/article/pii/S0010027711002022)

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