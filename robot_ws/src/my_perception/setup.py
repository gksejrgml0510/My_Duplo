from setuptools import setup
import os
from glob import glob

package_name = 'my_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # --- 이 아래 두 줄을 반드시 추가하세요! ---
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='han',
    maintainer_email='han@todo.todo',
    description='Duplo perception package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'duplo_perception_node = my_perception.duplo_perception_node:main'
        ],
    },
)