# **Deeplabv3-Pi-Test**

This repository contains a test implementation of the DeepLabV3-ResNet model for semantic segmentation on a Raspberry Pi. The project demonstrates running PyTorch's `deeplabv3_resnet50` model on limited hardware, including best practices for managing memory and system resources.

---

## **Table of Contents**
- [**Deeplabv3-Pi-Test**](#deeplabv3-pi-test)
  - [**Table of Contents**](#table-of-contents)
  - [**Installation**](#installation)
  - [**Setup**](#setup)
  - [**Running the Script**](#running-the-script)
  - [**Memory Management**](#memory-management)
  - [**Troubleshooting**](#troubleshooting)
  - [**Contributing**](#contributing)
  - [**License**](#license)
  - [**Acknowledgments**](#acknowledgments)
    - [**Commit Message:**](#commit-message)

---

## **Installation**

1. **Clone the Repository**
```bash
git clone https://github.com/Foley-ops/Deeplabv3-Pi-Test.git
cd Deeplabv3-Pi-Test
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Setup**

1. **Verify PyTorch & CUDA Installation**
```python
import torch
print(torch.cuda.is_available())  # Should return False on a Raspberry Pi
```

2. **Configure Swap Memory (Optional)**
If you encounter memory issues, create a swap file to increase available memory:

```bash
sudo fallocate -l 2G /swapfile  # Create a 2GB swap file
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

To make the swap file permanent, add the following line to `/etc/fstab`:
```plaintext
sudo echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

Check memory status:
```bash
free -h
```

---

## **Running the Script**

To run the `deeplabv3_single.py` script:
```bash
python3 deeplabv3_single.py
```

For running the script multiple times, run repeater.py, it's defaulted at 10, change the range.

```python
for i in range(10):
```

---

## **Memory Management**

1. **Garbage Collection:** The script uses `gc.collect()` to free up memory between runs.
2. **Clear CUDA Memory:**
```python
torch.cuda.empty_cache()
```
3. **Clear Linux Page Cache (Optional):**
```bash
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

## **Troubleshooting**

- **Out of Memory (OOM) Errors:**
  - Increase swap size or lower the model to `deeplabv3_resnet50`.
  - Monitor memory usage with:
  ```bash
  watch -n 1 free -h
  ```

- **Script Killed Unexpectedly:**
  - Ensure sufficient swap memory.
  - Lower input image resolution to reduce memory footprint:
  ```python
  input_image = input_image.resize((256, 256))
  ```

---

## **Contributing**

1. **Fork the repository**
2. **Create a feature branch:**
```bash
git checkout -b feature-branch
```
3. **Commit your changes:**
```bash
git commit -m "Description of changes"
```
4. **Push to the branch:**
```bash
git push origin feature-branch
```
5. **Create a Pull Request**

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## **Acknowledgments**

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- Raspberry Pi community for support and resources.

---

### **Commit Message:**
```bash
feat: Improve memory management and add swap file instructions

- Switched to deeplabv3_resnet50 to reduce memory usage
- Added garbage collection and memory clearing between runs
- Updated README with detailed instructions on configuring swap memory
- Added troubleshooting steps for Out of Memory (OOM) errors
```

