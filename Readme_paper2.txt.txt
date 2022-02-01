
go into folder octeract double click on 'octeract-engine-1.10.1-setup'

follow the 'Installation-Manual-_-Documentation_-Read-Online-_-Octeract' in the octeract folder 

The license file 'OcteractLicense_141_179_20220216.octeract_lic' is provided in the octeract folder.

setup the python environment using pyomo.yaml

please make changes shown in 'sol_edit.jpg' to sol.py which can be found at the location also shown in top of 'sol_edit.jpg'
i.e. users>your_username>miniconda3>envs>pyomo>Lib>site-packages>pyomo>opt>plugins>sol.py

In order for solver to work you need to be connected to the internet. 

set arrival time in task_set.csv ensure that arrival time for any task is between 0 - 85 seconds. 

run rhc.py

Example terminal outputs are shown in images 'example_octeract_output1-5'