from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'reactive_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zuriel_tov',
    maintainer_email='zuriel.tovar.m@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
            'console_scripts': [
            'reactive_navigation_node = reactive_navigation.reactive_navigation_node:main',
            'exploration_fsm_node     = reactive_navigation.exploration_fsm_node:main',
            'leg_detector_node        = reactive_navigation.leg_detector_node:main',
            'rrt_explorer_node        = reactive_navigation.rrt_explorer_node:main',
        ],
    },
)
