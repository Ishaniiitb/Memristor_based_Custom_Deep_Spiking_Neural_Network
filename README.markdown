# Memristor-based Custom Deep Spiking Neural Network

## 🧠 Description
This project implements a custom deep Spiking Neural Network (SNN) that leverages memristor characteristics for energy-efficient neuromorphic computing. The network is designed to mimic biological neural systems using a Leaky Integrate-and-Fire (LIF) neuron model and Spike-Timing-Dependent Plasticity (STDP) for learning. Memristors are used to simulate synaptic weights, enabling low-power and scalable hardware implementations. The project includes scripts to train and test the SNN on the MNIST dataset for handwritten digit recognition, along with utilities for data handling and visualization.

## 🚀 Features
- **Memristor-based Synapses**: Simulates synaptic weights using a memristor model for energy-efficient computing.
- **LIF Neuron Model**: Implements Leaky Integrate-and-Fire neurons for spiking behavior.
- **STDP Learning**: Uses Spike-Timing-Dependent Plasticity for unsupervised learning.
- **MNIST Classification**: Trains and tests the SNN on the MNIST dataset for digit recognition.
- **Visualization**: Includes tools to visualize network activity and performance metrics.
- **Modular Design**: Separates memristor, neuron, network, data handling, and visualization logic into distinct modules.

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ishaniiitb/Memristor_based_Custom_Deep_Spiking_Neural_Network.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Memristor_based_Custom_Deep_Spiking_Neural_Network
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include `numpy`, `matplotlib`, `torch`, and others listed in `requirements.txt`.

4. Ensure Python 3.8 or higher is installed.

## 📊 Usage
1. **Prepare Data**:
   - Run `data.py` to download and preprocess the MNIST dataset:
     ```bash
     python data.py
     ```

2. **Train the Model**:
   - Run `train.py` to train the SNN on the MNIST dataset:
     ```bash
     python train.py
     ```
   - Adjust hyperparameters (e.g., learning rate, epochs) in `train.py` as needed.

3. **Test the Model**:
   - Run `test.py` to evaluate the trained model on the MNIST test set:
     ```bash
     python test.py
     ```

4. **Visualize Results**:
   - Run `visualize.py` to generate plots of network activity and performance:
     ```bash
     python visualize.py
     ```

## 📂 File Structure
- `Memristor.py`: Defines the memristor model for synaptic weights.
- `SNN.py`: Implements the LIF neuron model and SNN architecture with STDP learning.
- `train.py`: Script for training the SNN on the MNIST dataset.
- `test.py`: Script for evaluating the trained SNN.
- `data.py`: Handles MNIST dataset loading and preprocessing.
- `visualize.py`: Generates visualizations of network activity and performance.
- `requirements.txt`: Lists required Python packages.

## 🛠 Technologies Used
- **Python**: Core programming language.
- **PyTorch**: Used for tensor operations and MNIST dataset handling.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualizing network activity and results.
- **Memristor Model**: Simulates synaptic weights inspired by memristor characteristics.
- **LIF Neurons and STDP**: Implements biologically inspired spiking neurons and learning rules.

## 👥 Contributors
Ishan Jha and Aryan Mishra

## 📚 References
- Inspired by research on memristor-based neuromorphic computing and spiking neural networks.[](https://www.sciencedirect.com/science/article/pii/S270947232400011X)[](https://ieeexplore.ieee.org/document/9806206/)[](https://www.researchgate.net/publication/384476302_Development_in_memristor-based_spiking_neural_network)
- Utilizes the MNIST dataset for benchmarking.
- Built with open-source libraries like PyTorch, NumPy, and Matplotlib.
