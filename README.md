# Brain Tumor Classification Web Application

A powerful deep learning application that leverages neural networks and the Gemini 1.5 Flash model to detect and classify brain tumors from MRI scans.

## Features

- Real-time brain tumor classification using neural networks
- Transfer learning models for improved accuracy
- Custom convolutional neural network architecture
- Integration with Gemini 1.5 Flash for detailed prediction explanations
- User-friendly Streamlit interface for easy interaction
- Comprehensive visualization of model predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PoPsMokE07/Brain-Tumor-Classification.git
cd brain-tumor-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root directory:
```bash
GEMINI_API_KEY=your_api_key_here
MODEL_PATH=path/to/saved/model
```

2. Place your trained model files in the specified directory.

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a brain MRI scan image through the interface

4. View the classification results and model explanations

## Project Structure

```
brain-tumor-classification/
├── app.py                 # Main Streamlit application
├── models/               # Neural network model definitions
│   ├── custom_cnn.py
│   └── transfer_learning.py
├── utils/               # Utility functions
│   ├── preprocessing.py
│   └── visualization.py
├── assets/             # Static assets
├── requirements.txt    # Project dependencies
└── README.md
```

## Model Architecture

The application uses two main neural network architectures:

1. Transfer Learning Model
   - Based on pre-trained networks
   - Fine-tuned for brain tumor classification
   - Optimized for medical imaging

2. Custom CNN
   - Specialized convolutional layers
   - Designed for MRI scan analysis
   - Trained on curated medical datasets

## Dependencies

- Python 3.8+
- Streamlit
- TensorFlow 2.x
- PyTorch
- OpenCV
- NumPy
- Pandas
- google.generativeai

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- Dataset provided by [[Source Name](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)]
- Based on research from [[Research Paper/Institution](https://medium.com/@bijil.subhash/explainable-ai-saliency-maps-89098e230100)]
