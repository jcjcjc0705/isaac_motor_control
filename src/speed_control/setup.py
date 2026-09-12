from setuptools import find_packages, setup

package_name = 'speed_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pochun',
    maintainer_email='pochun@todo.todo',
    description='Excitation-signal data collection for motor dynamics identification in Isaac Sim.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_collector = speed_control.data_collection:main',
            'gain_calibration = speed_control.limit_test:main',
        ],
    },
)
