<a id='index'></a>
# Deep Learning Projects

* [BreakingSimpleCaptchas](BreakingSimpleCaptchas) - a Convolutional Neural Network (CNN)-based captcha breaker
* [Autoencoder](Autoencoder) - a CNN-based autoencoder for denoising MNIST images
* [SmilesDetection](SmilesDetection) - real-time end-to-end CNN-based detection of smiles in a video stream
* [TextSentimentClassification](TextSentimentClassification) - sentiment classification in the IMDB movie reviews

## Setup

The steps should be followed in an order, i.e. setup TensorFlow first, then Jupyter, and finally install additional requirements.

* TensorFlow setup
  * [macOS (CPU)](#macos_tf)
  * [Ubuntu 16.04 LTS (CPU)](#ubuntu_cpu_tf)
  * [Ubuntu 16.04 LTS (GPU)](#ubuntu_gpu_tf)
* [Jupyter setup](#jupyter)
* graphviz and pydot setup
  * [macOS](#macos_graphviz)
  * [Ubuntu 16.04 LTS](#ubuntu_graphviz)
* Install additional requirements `pip install -r requirements.txt`

<a id='macos_tf'></a>
### TensorFlow setup on macOS (CPU)

  #### Get essentials
  * Install [brew](https://brew.sh/)
  * Run `brew install coreutils`
  * Install GCC `brew install gcc --without-multilib`
  * Install [anaconda and Python 3.6](https://www.anaconda.com/download/#macos)
  * Install [Java SE Development Kit 8](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

  #### Install Bazel
  ```
  brew install bazel && brew upgrade bazel 
  bazel version
  ```

  #### Setup a virtual Python environment
  * `conda create -n <your environment> python=3.6`
  * Activate the environment `source activate <your environment>`
  * Install Python dependencies `pip install six numpy wheel`

  #### Build TensorFlow from source
  * Clone 
    ```
    cd ~ && git clone https://github.com/tensorflow/tensorflow && cd ~/tensorflow
    ```
  * Configure the setup script `~/tensorflow/./configure` to use default values
  * Build TensorFlow with **CPU-only** support
    ```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    ```
  * Build Python wheel
    ```
    bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg
    ```
  * Install the wheel
    ```
    pip install tensorflow_pkg/*whl
    ```

  #### Test Tensorflow setup
  ```
  cd ~
  python
  import tensorflow as tf
  sess = tf.InteractiveSession()
  sess.close()
  ```

[back to top](#index) 


<a id='ubuntu_cpu_tf'></a>
### TensorFlow setup on Ubuntu 16.04 LTS (CPU)

#### Get essentials
```
sudo apt-get update 
sudo apt-get install build-essential software-properties-common -y 
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y 
sudo apt-get update && sudo apt-get install gcc-snapshot -y 
sudo apt-get update && sudo apt-get install gcc-7 g++-7 -y
```

#### Install Python and the required packages
```
sudo apt-get install git python-dev python3-dev build-essential swig libcurl3-dev libcupti-dev golang libjpeg-turbo8-dev make tmux htop cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev apt-transport-https ca-certificates curl software-properties-common coreutils mercurial libav-tools libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libtiff5-dev libx11-6 libx11-dev fluid-soundfont-gm timgm6mb-soundfont xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic fontconfig fonts-freefont-ttf wget unzip git nasm tar libbz2-dev libgtk2.0-dev libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev python-pip python3-pip python-wheel python3-wheel python-virtualenv 
```

#### Install Bazel
```
sudo add-apt-repository ppa:webupd8team/java 
sudo apt-get update && sudo apt-get install oracle-java8-installer
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add - 
sudo apt-get update 
sudo apt-get install bazel && sudo apt-get upgrade bazel
```

#### Setup a virtual Python environment
* `virtualenv -p python3 ~/<your environment>`
* Create an alias:
  ```
  echo "alias <your environment>='source ~/<your environment>/bin/activate'" >> ~/.bash_aliases
  ```
* Activate the environment `<your environment>`
* Upgrade pip `pip install --upgrade pip && pip install numpy opencv-python six wheel scipy`


#### Build TensorFlow from source
* Clone 
  ```
  cd ~ && git clone https://github.com/tensorflow/tensorflow && cd ~/tensorflow
  ```
* Configure the setup script ` ~/tensorflow/./configure`
* Build TensorFlow with **CPU-only** support
  ```
  bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
  ```
* Build Python wheel
  ```
  bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg
  ```
* Install the wheel
  ```
  pip install tensorflow_pkg/*whl
  ```

 #### Test Tensorflow setup
 ```
 cd ~
 python
 import tensorflow as tf
 sess = tf.InteractiveSession()
 sess.close()
 ```

 [back to top](#index) 

 <a id='ubuntu_gpu_tf'></a>
### TensorFlow setup on Ubuntu 16.04 LTS (GPU)

#### Get essentials
```
sudo apt-get update 
sudo apt-get install build-essential software-properties-common -y 
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y 
sudo apt-get update && sudo apt-get install gcc-snapshot -y 
sudo apt-get update && sudo apt-get install gcc-7 g++-7 -y
```

#### Disable the X11 GUI server
* Close all applications
* Press `ctrl + alt +f3`
* Login with your credentials
* Run `sudo service lightdm stop`

#### Disable the Nouveau kernel
* Create a new file `sudo nano /etc/modprobe.d/blacklist-nouveau.conf`
* Add the following lines:
  ```
  blacklist nouveau
  blacklist lbm-nouveau
  options nouveau modeset=0
  alias nouveau off
  alias lbm-nouveau off
  ```
* Update the initial RAM and reboot
  ```
  echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
  sudo update-initramfs -u
  sudo reboot
  ```

#### Install NVidia driver
* Run `sudo apt-get update && sudo apt-get install nvidia-396`
* `sudo reboot`
* If you're having problems logging in after the steps above, try disabling Fast Boot and UEFI Boot modes in your BIOS

#### Install CUDA 9.2
  ```
  cd ~/Downloads 
  wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 
  sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64.deb
  sudo apt-key add /var/cuda-repo-9.2-local/7fa2af80.pub
  sudo apt-get update
  sudo apt-get install cuda
  ```
#### Install cuDNN 7.14 Library for Linux
* Go to [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) 
* Download cuDNN 7.14
* Unzip `tar` file with `tar -xzvf cudnn-9.0-linux-x64-v7.tgz`
* Copy the following files into the CUDA Toolkit directory:
  ```
  sudo cp cuda/include/cudnn.h /usr/local/cuda/include
  sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
  sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn
  ```
 
#### Update path
```
echo $'export CUDA_HOME=/usr/local/cuda\nexport LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"\nexport PATH=$PATH:/usr/local/cuda/bin' >> ~/.bashrc`
source ~/.bashrc
```

#### Install Python and the required packages
```
sudo apt-get install git python-dev python3-dev build-essential swig libcurl3-dev libcupti-dev golang libjpeg-turbo8-dev make tmux htop cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev apt-transport-https ca-certificates curl software-properties-common coreutils mercurial libav-tools libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev libtiff5-dev libx11-6 libx11-dev fluid-soundfont-gm timgm6mb-soundfont xfonts-base xfonts-100dpi xfonts-75dpi xfonts-cyrillic fontconfig fonts-freefont-ttf wget unzip git nasm tar libbz2-dev libgtk2.0-dev libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev python-pip python3-pip python-wheel python3-wheel python-virtualenv 
```

#### Install Bazel
```
sudo add-apt-repository ppa:webupd8team/java 
sudo apt-get update && sudo apt-get install oracle-java8-installer
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add - 
sudo apt-get update 
sudo apt-get install bazel && sudo apt-get upgrade bazel
```

#### Setup a virtual Python environment
* `virtualenv -p python3 ~/<your environment>`
* Create an alias:
  ```
  echo "alias <your environment>='source ~/<your environment>/bin/activate'" >> ~/.bash_aliases
  ```
* Activate the environment `<your environment>`
* Upgrade pip `pip install --upgrade pip && pip install numpy opencv-python six wheel scipy`


#### Build TensorFlow from source
* Clone 
  ```
  cd ~ && git clone https://github.com/tensorflow/tensorflow && cd ~/tensorflow
  ```
* Configure the setup script ` ~/tensorflow/./configure` to use CUDA support (other options can be left to their default values)
* Build TensorFlow with GPU support
  ```
  bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
  ```
* Build Python wheel
  ```
  bazel-bin/tensorflow/tools/pip_package/build_pip_package tensorflow_pkg
  ```
* Install the wheel
  ```
  pip install tensorflow_pkg/*whl
  ```

 #### Test Tensorflow setup
 ```
 cd ~
 python
 import tensorflow as tf
 sess = tf.InteractiveSession()
 sess.close()
```

[back to top](#index) 

<a id='macos_r'></a>
### R and rpy2 setup on macOS

* Install [R](https://cran.r-project.org/)
* Find the path of `gcc` binary with `brew ls gcc`. It should be `/usr/local/Cellar/gcc/<version>/bin/gcc-<version>`.
* Export `gcc` path with `export CC=<path to gcc binary>` and `export CXX=<path to gcc binary>`.
* Install `rpy2` with `pip install rpy2`.

[back to top](#index) 

<a id='jupyter'></a>
### Jupyter setup

* Install Jupyter `pip install jupyter`
* Setup a separate Jupyter kernel for the virtual Python environment that was created above
    ```
    pip install ipykernel
    python -m ipykernel install --user --name <your-environment> --display-name "<your-environment>"
    ```

[back to top](#index) 

<a id='macos_graphviz'></a>
### graphviz and pydot setup on macOS

* Install graphviz `brew install graphviz`
* Install Python packages
  ```
  pip install graphviz==0.5.2
  pip install pydot-ng==1.0.0
  ```

[back to top](#index) 

<a id='ubuntu_graphviz'></a>
### graphviz and pydot setup on Ubuntu 16.04 LTS

* Install graphviz `sudo apt-get install graphviz`
* Install Python packages
  ```
  pip install graphviz==0.5.2
  pip install pydot-ng==1.0.0
  ```

[back to top](#index) 