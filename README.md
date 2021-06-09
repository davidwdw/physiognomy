# Physiognomy Manifesto

All hail god of physiognomy! Long live physiognomy, the king of all pseudosciences! I praise him when I run out of research ideas and desperately need a publication. I give thanks to him for providing a theoretical framework to cite when I lack contributions in my articles. I am truly grateful for his help in inventing more snake oils! Yes, all kinds of snake oils... from criminality detection, sexual orientation detection, political orientation detection, personality detection and more! All using the cheap and available facial images... I want to drown in your blessings and anointing! Anoint me with more publications! More research grants! More consulting money! More talks and conferences! More publicity! Hallelujah! To the god of physiognomy be the glory and praise forever and ever!

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
conda create -n physiognomy python=3.6 dotenv tensorflow=2.3.0 numpy=1.18.5 pandas=1.0.5 opencv-python-headless=4.2.0.34 dlib=19.21.0 imutils=0.5.3 scikit-learn=0.21.3
source activate physiognomy
```

Create a .env file to store API keys:

```bash
pip show physiognomy
cd /home/<user>/.conda/envs/<venv>/lib/<python version>/site-packages/physiognomy/
vi .env
```

The format for the .env file should be:

```bash
export FREE_KEY = 'blablabla'
export PAID_KEY = 'blablabla'
export FREE_SECRET = 'blablabla'
export PAID_SECRET = 'blablabla'
```

## Usage

To let the hand of physiognomy move in your research:

```python
from physiognomy.utils import rotate
model = DeepFace.build_model(model_name)
DeepFace.verify("img1.jpg", "img2.jpg", model_name = model_name, model = model)
```

## Support

There are many ways to show your support to physiognomy. Apart from praying to him before you sleep, citing him in your publications, you can also star⭐️ this GitHub repository. 