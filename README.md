# SuperCompose 🧩

This repo allows you to create objects using superquadrics. In particular we provide two version of the visualiser:
* gui.py: this version takes the rotation of the superquadric in the form of eulerian angle;
* gui_rot.py: this version takes the rotation of the superquadric in the form of a 3x3 rotation matrix.

It also enable you to load/download your superquadrics in a json format. An example of the json file (with a 3x3 rotation matrix) is provided in example.json. In save_json.py we provide some example code to generate the json file starting from superquadrics' prameters expressed as a numpy array.

You can install all the needed requirements by doing:
```bash
pip install -r requirements.txt
```