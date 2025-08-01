import os
from setuptools import setup, find_packages

exec(open('vcg/__version__.py').read())
os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
setup(
    name="vcg",
    packages=find_packages(exclude=["tests"]),
    version=__version__,
    author="Guoqing Zhang",
    license="MIT",
    python_requires='>=3.7', 
    include_package_data=True,
    entry_points={'console_scripts': [
        # train
        'train_vcg=vcg.train_vcg:main',    
        # test
        'test_vcg=vcg.test_vcg:main']
        },
)
