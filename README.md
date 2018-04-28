>non-official PyTorch implementation of VINet[1]

(Whole project are still under construction.)

# Installation

It's recommand to use docker image to run this project.
[Docker image installation guide](https://github.com/HTLife/VINet/wiki/Installation-Guide)

# Training
Log into container
```bash
sudo docker exec -it vinet bash
cd /notebooks/vinet
```

Execute main.py by
```bash
python3 main.py
```

# Note
## Network detail structure
<img src="https://user-images.githubusercontent.com/4699179/39375670-bc711a52-4a81-11e8-9be3-18b45924d0de.png" data-canonical-src="https://user-images.githubusercontent.com/4699179/39375670-bc711a52-4a81-11e8-9be3-18b45924d0de.png" width="400" />



[1] Clark, Ronald, et al. "VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem." AAAI. 2017.
