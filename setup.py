from setuptools import find_packages,setup
from typing import List


def get_requirements(filepath:str)->list[str]:
    '''This function provides the name of package'''
    requirement=[]
    with open(filepath) as object:
        requirement=object.readlines()
        requirement=[i.replace("\n","") for i in requirement]

        if '-e .' in requirement:
            requirement.remove('-e .')
    return requirement

setup(

    name = "project",
    version = "0.0.1",
    author = "Narendra",
    packages = find_packages(),
    install_requires = get_requirements("requirement.txt")
)