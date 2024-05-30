jmpy -i "."
cd ./dataml
#python pypi_install.py
pip install -r requirements.txt
cd ..
python setup.py bdist_wheel