from setuptools import setup, find_packages

setup(
    name = "gaspery",
    version = "0.2.2", 
    #packages = find_packages(),
    packages = ['gaspery'],
    package_dir={"":"src"},
    author = 'Chris Lam',
    author_email = 'c.lam@ufl.edu',
    description = 'Fisher Information-based radial velocity observation scheduling',
    license = 'MIT License'
    #install_requires = [
    #    'tinygp',
    #    'numpy'
    #]
)
