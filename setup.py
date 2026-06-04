from setuptools import setup, find_packages

setup(
    name='pose6d',
    version='0.1.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        # Minimal, user should install full requirements.txt
    ],
    include_package_data=True,
    description='6D object pose estimation project for robot grasping',
)
