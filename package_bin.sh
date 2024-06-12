cd ./dataml
#python pypi_install.py
pip install -r requirements.txt
cd ..
cp -r dataml dist/
jmpy -i "."
cp setup.py ./dist/dataml/
cd ./dist/dataml
python setup.py bdist_wheel
cd ..
#python setup.py install

