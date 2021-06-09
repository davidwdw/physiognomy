from setuptools import setup

setup(
    name='physiognomy',
    version='0.1.0',    
    description='All hail god of physiognomy! Long live the pseudoscience of physiognomy!',
    url='https://github.com/davidwdw/physiognomy',
    author='Dawei Wang',
    author_email='david.wang@kellogg.northwestern.edu',
    license='BSD 2-clause',
    packages=['physiognomy'],
    install_requires=['numpy'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
    ],
)