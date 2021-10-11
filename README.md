## Setup


1. Install Python 3.7.9. (https://realpython.com/installing-python/)

    > This version of Python as well as the other packages below are intentionally selected for compatibility.

2. If you have not installed `virtualenv`, install it using the command `python -m pip install virtualenv`.

3. Create a virtual environment using the command `python -m virtualenv venv`.

4. Activate the virtual environment by running `venv\Scripts\activate` on Windows or `. venv/bin/activate` on MacOS or Linux.

5. Install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki), adding the install location in your `path` environment variable. (Don't forget to restart your command prompt / terminal, and re-activate your virtual environment)

6. Modify the `settings.ini` file with the path Tesseract is installed in. The path should end in "tessdata".

7. Install `pytorch` using the command `python -m pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`

8. Install `tesserocr` by following the instructions under "Windows" here (https://pypi.org/project/tesserocr/).
    - download the .whl file of the latest compatible version here (https://github.com/simonflueckiger/tesserocr-windows_build/releases)
    - install the .whl file by using `python -m pip install <path-to-whl>.whl`

9. Install the pip dependencies by running `python -m pip install -r requirements.txt`.

10. To test if everything is working correctly, run the command `python .\main.py`, which should create a folder named "censored", populated with censored versions of the images found in the "image" folder.
