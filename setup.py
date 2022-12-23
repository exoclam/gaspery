from setuptools import setup, find_packages
setup(
    name = "gaspery",
    version = "0.1.1", 
    packages = find_packages(),
    url = 'https://github.com/cl3425/gaspery',
    author = 'Chris Lam',
    author_email = 'c.lam@ufl.edu',
    license = 'MIT',
    install_requires = [
        'celerite2',
        'numpy'
    ]
)