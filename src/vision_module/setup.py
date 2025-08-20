from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision_module'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        # 這些路徑是相對於 vision_module/ 目錄
        'vision_module': [
            'template/*.png',
            'template2_shape/*.png',
            'template_l_shape/*.png',
        ]
    },
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ✅ 新增這三行到 share
        ('share/' + package_name + '/template', glob('vision_module/template/*.png')),
        ('share/' + package_name + '/template2_shape', glob('vision_module/template2_shape/*.png')),
        ('share/' + package_name + '/template_l_shape', glob('vision_module/template_l_shape/*.png')),

        # 你原本裝到 lib 的可留可不留
        (os.path.join('lib', package_name, 'template'), glob('vision_module/template/*.png')),
        (os.path.join('lib', package_name, 'template2_shape'), glob('vision_module/template2_shape/*.png')),
        (os.path.join('lib', package_name, 'template_l_shape'), glob('vision_module/template_l_shape/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='henry',
    maintainer_email='HENRY.HY.CHANG@deltaww.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'realsense_rough = vision_module.realsense_rough:main',
            'realsenselib = vision_module.realsenselib:main',
            'detection_node = vision_module.detection_node:main',
            'screw_detector = vision_module.screw_detector:main',
            'l_shape_detector = vision_module.l_shape_detector:main',
            'icp_fitter = vision_module.icp_fitter:main',
            'realsense_capture_node = vision_module.realsense_capture_node:main',
            'roi_preview_node = vision_module.roi_preview_node:main',
            'dectionL_node = vision_module.dectionL_node:main',
        ],
    },
)
