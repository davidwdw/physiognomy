# Physiognomy Manifesto

All hail god of physiognomy! Long live physiognomy! May the king of pseudosciences reign for a thousand years! <br>

Physiognomy is a faith, an art, a religion, a belief and a perspective to life itself. It tells us that the only truth in behaviors lies in our faces [`Wiki definition`](https://en.wikipedia.org/wiki/Physiognomy). Yes, because faces are windows to our souls. They reflect who we are, what we are, and where we will be. We can use faces to predict the future! You can refer to the old testament of physiognomy, published in 1800s by the prophetic [`Cesare Lombroso`](https://archive.org/details/criminalmanaccor1911lomb/mode/2up?view=theater). <br>

<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b4/The_relation_between_the_human_physiognomy_and_that_of_the_Wellcome_L0010074.jpg" width="95%" height="95%"></p>

A revival of physiognomy is happening. So many new believers in recent years! Particularly, folks in the field of machine learning and computer vision. Long gone are the days that physiognomists get persecuted and have to hide their faith away from the world!<br> 

Pray to physiognomy when you run out of research ideas. Pray to physiognomy when you desperately need a publication. Thank physiognomy when it provides you a pseudoscientific theoretical framework to use when you need to write about psychology. Now you can easily publish articles about psychology in machine learning journals! What a way to go!<br> 

We are all truly grateful for modern physiognomy's help in inventing [`snake oils`](https://www.cs.princeton.edu/~arvindn/talks/MIT-STS-AI-snakeoil.pdf)! Yes, all kinds of snake oils... from criminality detection, sexual orientation detection, personality detection and more! All using the omnipresent sources of facial images found on social media platforms.<br>

Let us drench our souls in these snake oils. Yes, in these blessings and anointings! Anoint us with more publications! More research grants! More consulting money! More talks! More conference papers! <br> 

To the faith of physiognomy be the glory and praise forever and ever!<br> 

#### Declarations of physiognomy in machine learning
- [`Criminality detection`](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0282-4)
- [`Criminality detection`](https://arxiv.org/pdf/1611.04135.pdf)
- [`Personality detection`](https://link.springer.com/content/pdf/10.1007/s10462-019-09770-z.pdf)
- [`Personality detection`](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9244051)
- [`Personality detection`](https://hal.inria.fr/hal-01677962/file/apparent_personality.pdf)

#### Testaments of physiognomy by social scientists
- [`Leader Emergence detection`](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159950)
- [`Sexual orientation prediction`](https://psycnet.apa.org/record/2018-03783-002)
- [`Personality detection`](https://www.nature.com/articles/s41598-017-00071-5)
- [`The face of risk`](https://www.researchgate.net/publication/323997454_The_face_of_risk_CEO_facial_masculinity_and_firm_risk)
- [`The face of success`](https://journals.sagepub.com/doi/10.1111/j.1467-9280.2008.02054.x)
- [`CEO enumeration`](https://www.sciencedirect.com/science/article/pii/S0148296320302654)
- [`CEO leadership`](https://www.researchgate.net/publication/288701149_The_big_man_has_a_big_mouth_Mouth_width_correlates_with_perceived_leadership_ability_and_actual_leadership_performance)

#### Blasphemies against physiognomy by computer scientists
- [`Limits to predicatbility`](https://dl.acm.org/doi/10.1145/2872427.2883001)
- [`Shortcut learning`](https://www.nature.com/articles/s42256-020-00257-z)
- [`Physiognomy???s new clothes`](https://medium.com/@blaisea/physiognomys-new-clothes-f2d4b59fdd6a)
- [`Self-presentation`](https://medium.com/@blaisea/do-algorithms-reveal-sexual-orientation-or-just-expose-our-stereotypes-d998fafdf477)

#### Counterarguments by social scientists
- [`Stable face representations`](https://royalsocietypublishing.org/doi/full/10.1098/rstb.2010.0379)
- [`Variability of facial images`](https://www.sciencedirect.com/science/article/pii/S0010027711002022)
- [`Facial width height ratios`](https://journals.sagepub.com/doi/full/10.1177/0956797617716929)
- [`Facial width height ratios`](https://journals.sagepub.com/doi/full/10.1177/0956797619849928)

#### Books and other resources on this matter
- [`Calling Bullshit`](https://www.callingbullshit.org)

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

Apart from praying to physiognomy before you sleep, citing it in your publications, you can also star?????? this GitHub repository. Let physiognomy reign forever and ever!