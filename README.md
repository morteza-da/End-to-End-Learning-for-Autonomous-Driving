# End-to-End Learning for autonomous-driving
This project presents an end-to-end deep learning model developed for decision-making in autonomous vehicles. It is based on my Ph.D. dissertation, integrating LiDAR and camera inputs to predict driving actions using CNN, LSTM, and Attention mechanisms.

## Model Overview
**Input**: LiDAR point clouds + Camera images
**Architecture**: Dual CNN (Xception & ResNeXt) → LSTM + Attention
**Output**: Steering angle and speed commands
**Method**: End-to-End learning (E2E) on real-world data

## 🗂️ Project Structure
├── data/ # Sample input data (or links to it)
├── src/ # Source code
├── models/ # Trained models (optional or linked)
├── notebooks/ # Jupyter notebooks for analysis
├── results/ # Visualizations, graphs, and logs
└── README.md 

 ## 📊 Dataset
- 11,000+ image frames + synchronized LiDAR data
- Real-world driving scenarios (urban & highway)
- Pre-trained backbones: ImageNet (for CNN feature extractors)

- ## ⚙️ Dependencies
- Python 3.10+
- TensorFlow / PyTorch
- OpenCV, NumPy, Pandas
- Matplotlib / Seaborn

- ## 🚀 How to Run
```bash
git clone https://github.com/yourusername/e2e-autonomous-driving.git
cd e2e-autonomous-driving
pip install -r requirements.txt
python src/train.py

## 📈 Results
- Training RMSE: ...
- Validation RMSE: ...
- Test RMSE: ...

## 🔍 Innovation Highlights
- First-time combination of [Xception + ResNeXt + LSTM + Attention] in E2E driving
- Parallel CNN feature extraction from multi-modal data
- Real-world deployment-ready structure

## 🧑‍💻 Author
**[Morteza Aghamohammadi]**  
Ph.D. in Mechanical Engineering – Autonomous Vehicles  
[LinkedIn](https://www.linkedin.com/in/morteza-a/)) | [Email](aghamohamadi.da@gmail.com) | [ORCID](0009-0001-4837-2298)
