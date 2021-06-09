from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='physiognomy',
    version='0.1.0',    
    description='All hail god of physiognomy! Long live the pseudoscience!',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/davidwdw/physiognomy',
    project_urls={
        "Bug Tracker": "https://github.com/davidwdw/physiognomy/issues",
    },
    author='Dawei Wang',
    author_email='david.wang@kellogg.northwestern.edu',
    license='BSD 2-clause',
    packages=['physiognomy'],
    install_requires=['numpy',
                      python=3.6 tensorflow=2.3.0 numpy=1.18.5 pandas=1.0.5 opencv-python-headless=4.2.0.34 dlib=19.21.0 imutils=0.5.3 scikit-learn=0.21.3],

    classifiers=[
        'Development Status :: 1 - Testing',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',  
        'Programming Language :: Python :: 3.6',
    ],
)