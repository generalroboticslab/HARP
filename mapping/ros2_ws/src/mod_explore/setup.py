from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mod_explore'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    package_data={'mod_explore': ['map_editor.html']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.viz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.rviz')),
        # (os.path.join('share', package_name, 'launch', 'duke_traj'), glob('launch/duke_traj/*.csv')),
        # Avoid glob('launch/*'): it matches __pycache__ and breaks install (dirs are not files).
    ],
    install_requires=[],
    zip_safe=True,
    maintainer='pattylo',
    maintainer_email='you@example.com',
    description='explore mo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'explore_zigzag = mod_explore.explore_zigzag:main',
            'explore_viz = mod_explore.explore_viz:main',
            'map_recorder = mod_explore.map_recorder:main',
        ],
    },
)
