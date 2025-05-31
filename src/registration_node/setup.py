from setuptools import find_packages, setup

package_name = 'registration_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yifanwang',
    maintainer_email='yifanwang@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'object_registration= registration_node.object_registration:main',
            'pcd_segmentation = registration_node.pcd_segmentation:main',
    },
)
