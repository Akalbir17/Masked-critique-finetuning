from setuptools import setup, find_packages

setup(
    name='masked-critique-finetuning',
    version='0.1.0',
    packages=find_packages(include=['btsft', 'btsft.*']),
    package_data={
        'btsft': ['config/*.yaml'],
    },
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'bitsandbytes>=0.39.0',
        'unsloth>=0.1.0',
        'pyyaml>=6.0',        
        'typing-extensions',   
        'tqdm',               
        'wandb',              
        'numpy>=1.24.0',      
        'accelerate>=0.20.0', 
        'tensorboard',        
    ],
    entry_points={
        'console_scripts': [
            'mcf=btsft.main:main',
        ],
    },
    python_requires='>=3.8',
    author='Akalbir Singh Chadha and Chandresh Mallick',
    description='Masked Critique Fine-tuning for Small Reasoning Models',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/masked-critique-finetuning',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)