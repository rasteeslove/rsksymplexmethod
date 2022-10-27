from distutils.core import setup

setup(
  name = 'rsksymplexmethod',
  packages = ['rsksymplexmethod'],
  version = '0.1',
  license='MIT',
  description = 'Simplex method by rasteeslove.',
  author = 'Rostislav Kayko',
  author_email = 'krastsislau@gmail.com',
  url = 'https://github.com/rasteeslove/rsksymplexmethod',
  download_url = 'https://github.com/rasteeslove/rsksymplexmethod/archive/refs/tags/v_01.tar.gz',
  keywords = ['Math', 'Simplex', 'Method'],
  install_requires=[
          'numpy',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.10',
  ],
)
