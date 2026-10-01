from glob import glob

from setuptools import setup

PACKAGE = "litert_pose_estimation_demo"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE]),
        ("share/" + PACKAGE, ["package.xml"]),
        ("share/" + PACKAGE + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Julius Kammerl",
    maintainer_email="julius@kammerl.de",
    description="Publishes an RGB-D frame and triggers LiteRT pose estimation.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "rgbd_publisher = litert_pose_estimation_demo.rgbd_publisher:main",
        ],
    },
)
