# PRIVACY-PRESERVING VIDEO CLASSIFICATION with CNNs
--------------------

## Overview
This project provides a baseline approach for an end-to-end privacy-preserving video classification pipeline which involves three major steps: oblivious selection of frames in a video, securely classifying individual frames in the video using existing protocols for image classification, and secure label aggregation across frames to obtain a single class label for the video. To this end, we use Secure MultiParty Computation (MPC) for privacy-preserving single frame based video classification with a pre-trained convolutional neural network for image classification while keeping the parameters of the CNN model and the contents of the video private. We demonstrate our approach for the use-case of private human emotion recognition where videos can be classified with state-of-the-art accuracy, and without leaking sensitive user information. Our proposed solution is evaluated across various security settings - honest and dishonest majority with both active and passive adversaries. 

##### Motivation
Video classification using deep learning is widely used in many applications such as facial recognition, gesture analysis, activity recognition, emotion analysis, etc. Many applications capture user videos for classification purposes. This raises concerns regarding privacy and potential misuse of videos. As videos are available on the internet or with the service providers, it is possible to misuse them such as to generate fake videos or to mine information from the videos that goes beyond the professed scope of the original application or service. The service provider on the other hand typically cannot provide the trained video classification model to be run on the client's side either, due to resource constraints, proprietary concerns and the risk for adversarial attacks. There is a need for technology to perform video classification in a privacy-preserving manner, i.e. such that the client does not have to share their videos with anyone without encryption, and the service provider does not have to show their model parameters in plaintext.

--------------------
## Project Structure
##### In-the-clear (without encryption)
- trainCNN.py - Script to preprocess videos and fine tune the CNN model for RAVDESS dataset
- inferVideo.py - Script to preprocess videos and infer all test videos
- prepDataForSecure.py - Script to extract the parameters of the model and the contents of the video separately for inputs to privacy preserving framework.
- GlobalVars.py - Declares all global constants for the project in-the-clear
- models/facial_expression_model_weights.h5 - Weights after training with FER 2013 dataset
- models/model_to_json.json - Architecture of the CNN used to train with FER 2013 dataset
- models/trainedAct.h5 - Fine tuned model on RAVDESS dataset. Final model used for evaluation of privacy preserving inference.

##### In privacy-preserving environment
- PPVC.mpc - Script to privately classify a video 
- ml.py - Modifications made to an exisiting file in MP-SPDZ
- sample_video.txt - Extracted contents of a video (Refered to as V in the paper i.e A in the Section 4.1 in the paper)
- sample_parameters.txt - Extracted contents to classify sample_video (Includes CNN model M and Frame selection matrix B as referred in the paper)

## DataSets
- The CNN is trained using [FER 2013](https://datarepository.wolframcloud.com/resources/FER-2013) dataset
- The CNN is then fine-tuned using [RAVDESS](https://zenodo.org/record/1188976) dataset*
- The evaluation of the video classification (both in-the-clear and private) is done on [RAVDESS](https://zenodo.org/record/1188976) dataset*

* For further information on dataset and splits, please refer to the Section 5.1 in the paper.
--------------------


## Installing requirements
##### Installing requirements for experiments in-the-clear

- Python 3
- TensorFlow 2.x
```sh
pip install tensorflow-gpu
# pip install tensorflow
```
- Keras
```sh
$ pip install Keras
```
- Keract 
```sh
$ pip install keract
```
- MTCNN
```sh
$ pip install mtcnn
```
- OpenCV2
```sh
$ pip install opencv-python==2.4.9
```

##### Installing requirements for privacy-preserving environment
Open Source MPC based framework: [MP-SPDZ](https://https://github.com/data61/MP-SPDZ) and its prerequisites as mentioned in the repository
>Requirements
-GCC 5 or later (tested with up to 10) or LLVM/clang 5 or later (tested with up to 11). We recommend clang because it performs better.
-MPIR library, compiled with C++ support (use flag --enable-cxx when running configure). You can use make -j8 tldr to install it locally.
-libsodium library, tested against 1.0.16
-OpenSSL, tested against 1.1.1
-Boost.Asio with SSL support (libboost-dev on Ubuntu), tested against 1.65
-Boost.Thread for BMR (libboost-thread-dev on Ubuntu), tested against 1.65
-64-bit CPU
-Python 3.5 or later
-NTL library for homomorphic encryption (optional; tested with NTL 10.5)
-If using macOS, Sierra or late
- Installing general prerequisites
```sh
$ apt-get install automake build-essential git libboost-dev libboost-thread-dev libsodium-dev libssl-dev libtool m4 python texinfo yasm
```
- Installing and configuring MPIR Library
```sh
$ wget http://mpir.org/mpir-3.0.0.zip
$ unzip mpir-3.0.0.zip
$ cd mpir-3.0.0
$ ./configure --enable-cxx
$ make
$ make check
$ sudo make install
$ sudo ldconfig
```
- Installing and configuring clang 9.0 on ubuntu 18.04
```sh
$ sudo apt-get install build-essential xz-utils curl
$ curl -SL http://releases.llvm.org/9.0.0/clang%2bllvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz | tar -xJC .
$ mv clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 clang_9.0.0
$ sudo mv clang_9.0.0 /usr/local
$ export PATH=/usr/local/clang_9.0.0/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/clang_9.0.0/lib:$LD_LIBRARY_PATH
```

--------------------
## Preparing and extracting data in-the-clear
Note : Update GlobalVars.py for folder location
##### Step 1: Train the model
Run trainCNN.py
```sh
python3 trainCNN.py
```
##### Step 2: Evaluate the model on test dataset
Run inferVideo.py
```sh
python3 inferVideo.py
```
##### Step 3: Extract the model parameters and contents of the video to be inferred, setup the frame selection matrix
- Run prepDataForSecure.py
```sh
python3 prepDataForSecure.py
```
After running the above script, two files will be generated Input-P1-0 (this holds the parameters of the CNN model for frame classification) and Input-P0-0 (this holds the pixel valus of all the frames in the video)

- Append at the beginning the frame selection matrix to Input-P1-0*: Preprocess the video to get the total number of frames N. Generate a zero-matrix of size [N/15 , N]. In every ith row, change the value of (15*i)th column to 1. Copy the contents of the generated matrix in row-major order to Input-P1-0. 

* For further information, please refer to Section 4.1 in paper.

## Setting up environment for privacy-preserving inference
##### Step 1: Setting up virtual machines
- Launch required number of virtual machines (based on the protocol to be used) in the cloud with Ubuntu as the operating system and with ssh access. For our experiments, we launched F32s V2 VMs on MS Azure. For example, if experimenting with 3PC, launch 3 similar virtual machines also called as parties named as P0, P1 and P2.
- Setup the inbound and outbound rules for each virtual machine (eg: enable TCP port 4999-5000) to enable communication between these machines for MPC protocols.
- Update and upgrade if necessary
```sh
$ sudo apt update
$ sudo apt upgrade
```
- Install all the [requirements](#installing-requirements-for-privacy-preserving-environment) for [MP-SPDZ](https://https://github.com/data61/MP-SPDZ) 
- Clone the [MP-SPDZ](https://https://github.com/data61/MP-SPDZ) framework on each machine.
```sh
$ git clone https://github.com/data61/MP-SPDZ.git  
```
- Compile the framework on each virtual machine as described in the README of MP-SPDZ repository. For our experiments, we used the follwoing protocols and corresponding scripts from the framework and compiled the framework accordingly.
```sh
$ make semi2k-party.x spdz2k-party.x replicated-ring-party.x sy-rep-ring-party.x rep4-ring-party.x   
```
|Security Setting | Protocol | Protocol-Program |
| ------ | ------ | ------ |
| Dishonest majority and Semi-honest (2PC) | SPDZ2k | semi2k-party.x |
| Dishonest majority and Malicious (2PC) | OTSemi2k | spdz2k-party.x |
| Honest majority and Semi-honest (3PC) | Replicated2k | replicated-ring-party.x |
| Honest majority and Malicious (3PC) | SPDZwise-Replicated2k | sy-rep-ring-party.x |
| Honest majority and Malicious (4PC) | Rep4-2k | rep4-ring-party.x|


- Create a directory named 'Player-Data' on each virtual machine in the MP-SPDZ directory 
```sh
$ cd MP-SPDZ
$ mkdir Player-Data
```
- Set up an ssl connection for the required number of parties. For our experiments, we set-up ssl connection for 4 parties. Run the below command on a single virtual machine, P0. Make sure that MP-SPDZ is the present working directory.
```sh
$ pwd
$ Scripts/setup-ssl.sh 4
```
The above script will generate .key and .pem files for 4 parties in the folder Player-Data. Copy the .pem files to  and distribute the corresponding .key files to the MP-SPDZ/Player-Data/ folder of all the virtual machines such that P0 must hold only P0.key and P0.pem,P1.pem,P2.pem and P3.pem in the Player-Data.
- After the keys and certificates are correctly ditributed and shared among all parties in the MP-SPDZ/Player-Data/ folder, run the below command on all virtual machines from the MP-SPDZ directory so that ssl recongnzes .key and .pem files during communication.
```sh
$ c_rehash Player-Data/
```
- Create a file called HOSTS on each virtual machine by in the the MP-SPDZ directory. Write the IP addresses of all computing machines o this file.
```sh
eg: vi HOSTS
10.0.0.4
10.0.0.5
10.0.0.6
10.0.0.7
```
- Copy PPVC.mpc to MP-SPDZ/Programs/Source/ of all the virtual machines.
- Copy ml.py to MP-SPDZ/Compiler/ of all the virtual machines.

##### Step 2: Copying private data on virtual machines 
- Copy the generated Input-P0-0 to the MP-SPDZ/Player-Data/ of P0. The sample_video.txt provided can be renamed as Input-P0-0 for testing.
- Copy the generated Input-P1-0 to the MP-SPDZ/Player-Data/ of P1. The sample_parameters.txt provided can be renamed as Input-P1-0 for testing.
Through out our experiments, we assume P0 as the party who initiates the program and the final class label of the video is revealed to P0. 
--------------------

## Running privacy-preserving video inference for a video from RAVDESS dataset
- Edit the required parameters in MP-SPDZ/Programs/Source/PPVC.mpc on each virtual machine such as Layer.n_threads, N as the total number of frames in the video and n as the number of selected frames in the video. Eg: Layer.n_threads = 32, N = 92, n = 7.  
- Compile the MP-SPDZ/Programs/Source/PPVC.mpc on all virtual machines. Make sure MP-SPDZ is the present working directory.
 ```sh
Without mixed circuits
$ ./compile.py -R 64 PPVC.mpc
With mixed circuits - local share coversion for replicated secret sharing based schemes
$ ./compile.py -Z <num_parties> -R 64 PPVC.mpc
With mixed circuits - edaBits for 2PC (dishonest majority)
$ ./compile.py -Y  -R 64 PPVC.mpc
```
The above command will enable MPC protocols to execute in the ring domain (2^k) with k = 64.
- On each virtual machine execute the following command template
 ```sh
$ ./<protocol-program>.x <machine_id> -R 64 PPVC -pn 5000 -h <IP of P0>
eg: On P1 for honest majority,semi-honest (3PC)
$ ./replicated-ring-party.x P1 -R 64 PPVC -pn 5000 -h <IP of P0>
```
To run with mixed circuit computations for 2PC with malicious adversary add -B 4 
eg: On P1 for dishonest majority active (2PC), run using following command
 ```sh
$ ./spdz2k-party.x P1 -B 4 -R 64 PPVC -pn 5000 -h <IP of P0>
```
- After execution of the program, the class label will be revealed to P0.



