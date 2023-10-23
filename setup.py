from setuptools import find_packages,setup
from typing import List

def get_req(filepath:str)->List[str]:

    requirements=[]
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    #print(requirements)
    return requirements



setup(
name='ML end to end project',
version='0.0.1',
author='Chirag',
author_email='mahajan.chirag3@gmail.com',
packages=find_packages(),
install_requires=get_req("requirements.txt")

)