from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'valet_parking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sechankim',
    maintainer_email='sechankim98@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'canny_parking_done_node = valet_parking.canny_parking_done_node:main',
            'cmd_stop_mux_node = valet_parking.cmd_stop_mux_node:main',
            'collector_node = valet_parking.collector_node:main',        
            'collector_stop_node = valet_parking.collector_stop_node:main',   
            'dagger_controller = valet_parking.dagger_controller:main',
            'depth_det_node = valet_parking.depth_det_node:main',
            'det_stop_node = valet_parking.det_stop_node:main',            
            'inference_multitask_node_total = valet_parking.inference_multitask_node_total:main',
            'multitask_infer_node = valet_parking.multitask_infer_node:main',
            'parking_done_voter_node = valet_parking.parking_done_voter_node:main',
            'policy_infer_node = valet_parking.policy_infer_node:main',
            'seg_done_node = valet_parking.seg_done_node:main',
            'seg_line_yolo = valet_parking.seg_line_yolo:main',
            'stop_controller = valet_parking.stop_controller:main',
        ],
    },
)
