import torch

# Replace with the actual file path if necessary
model_path = "deploy/pre_train/g1/stand_still.pt"
try:
    model = torch.jit.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")