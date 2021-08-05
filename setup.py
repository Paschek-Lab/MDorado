from setuptools import find_packages, setup

setup(
	name='mdorado',
	packages=find_packages(include=['mdorado']),
	version='0.1.1',
	python_requires='>=3',
	description='Collection of scripts to analyse molecular dynamics simulations.',
	author='Jan Neumann',
	license='GPL-3.0',
        install_requires=[
            'numpy>=1.20.0',
            'scipy>=1.4.0',
            'MDAnalysis==1.1.1',
            ],
        package_data={
            "mdorado": ["data/*"],
            },
)
