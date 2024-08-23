from setuptools import find_packages, setup
from os import path

version_file = "yolov7/version.py"


if __name__ == "__main__":
    setup(
        name="yolov7",
        description="YOLOv7",
        packages=find_packages(exclude=("configs", "tools", "demo", "images")),
        classifiers=[
            "Programming Language :: Python :: 3.11",
        ],
        license="Apache License 2.0",
        zip_safe=False,
    )