from setuptools import setup

setup(
    name="template_experiment",
    version=0.1,
    description="...",
    url="http://github.com/fynnbe/template_experiment",
    author="Fynn Beuttenmüller",
    author_email="thefynnbe@gmail.com",
    license="Apache License 2.0",
    packages=["template_experiment"],
    install_requires=[
        "torch=1.1.0",
        "dill=0.2.9",
        "pyyaml=5.1",
        "scipy=1.2.1",
        "h5py=2.9.0",
        "numpy=1.16.3",
        "scikit-image=0.15.0",
    ],
    zip_safe=False,
)
