from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['imagineer'],
    scripts=['scr/ai_service']
    package_dir={'': 'src'}
    )

setup(**d)
