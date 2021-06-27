import io,os
from setuptools import setup

def readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(
            os.path.join(base_dir, 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()

install_requirements = [
    "requests",
    "scipy",
    "numpy",
    "pandas",
    "opencv-python-headless",
    "scikit-learn",
    "matplotlib",
    "python-dotenv",
    "statsmodels",
]
    
setup(
    name='physiognomy',
    version='0.2.2',    
    description='All hail god of physiognomy! Long live the pseudoscience!',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/davidwdw/physiognomy',
    author='Dawei Wang',
    author_email='david.wang@kellogg.northwestern.edu',
    license='BSD 2-clause',
    packages=['physiognomy'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',  
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=install_requirements,
)
