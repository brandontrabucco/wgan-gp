from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['tensorflow==2.1.2',
                     'tensorflow-probability',
                     'tensorflow-datasets',
                     'tensorflow-addons',
                     'numpy',
                     'jupyter',
                     'matplotlib']


PACKAGES = [package
            for package in find_packages() if
            package.startswith('wgangp')]


setup(name='wgangp',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=PACKAGES,
      description='Training Wasserstein GAN with Gradient Penalties')
