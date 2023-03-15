from setuptools import setup, find_packages

setup(
    name='stable_diffusion', # replace with the name of your package
    version='0.1', # replace with the version of your package
    packages=find_packages(),
    install_requires=[
    ],
    author='Malin Braatz, Mathilda Buschmann, Alice Schirrmeister',
    author_email='mbraatz@uni-osnabrueck.de, mbuschmann@uni-osnabrueck.de, aschirrmeist@uni-osnabrueck.de',
    description='Description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='hhttps://github.com/tensorflow-project/stable-diffusion',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
)
