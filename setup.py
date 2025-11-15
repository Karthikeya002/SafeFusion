from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='safefusion',
    version='1.0.0',
    author='Dr. T. Kalaichelvi, Derangula Alekhya, K. Karthikeya, V. S. Ramakrishna',
    author_email='vtu22893@veltech.edu.in',
    description='A YOLO-Transformer Hybrid Model for Intelligent Accident Surveillance',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Karthikeya002/SafeFusion',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.2.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.990',
        ],
    },
    entry_points={
        'console_scripts': [
            'safefusion=src.safefusion:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
