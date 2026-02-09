from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'valet_parking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='11306260+liangfuyuan@user.noreply.gitee.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publish = valet_parking.camera_publish:main',
            'cmd_mux_node = valet_parking.cmd_mux_node:main',
            'joystick_node = valet_parking.joystick_node:main',
            'policy_node = valet_parking.policy_node:main',
            'cam_publish_for_dataset = valet_parking.cam_publish_for_dataset:main',
            'cmd_bridge_mux_node = valet_parking.cmd_bridge_mux_node:main',      
            'joystick_bridge_node = valet_parking.joystick_bridge_node:main',
        ],
    },
)

