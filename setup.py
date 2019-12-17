from setuptools import setup

setup(
    name='easing',
    version='0.1.0',
    packages=['easing'],
    license='MIT',
    install_requires=[
    'matplotlib>=3.0.3',
    'numpy>=1.16.1',
    'pandas>=0.24.1'
    ],
    url='https://github.com/NicholasARossi/Easing-Animations-with-Python',
    download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',
    author='Nicholas A. Rossi',
    author_email='nicholas.rossi2@gmail.com',
    description='Generating smooth animations in python',
    keywords = ['ANIMATION', 'SMOOTH', 'VISUALIZATION'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)


