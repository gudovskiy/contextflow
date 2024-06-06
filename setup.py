from setuptools import setup

setup(
    name='ContextFlow',
    version='1.0',
    description="ContextFlow",
    author="",
    author_email='',
    packages=[
        'contextflow'
    ],
    entry_points={
        'console_scripts': [
            'contextflow=contextflow.cli:main',
        ]
    },
    python_requires='>=3.6',
)
