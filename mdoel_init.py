from models.experimental import attempt_load
import torch
from plate_recognition.plate_rec import init_model


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detect_model = load_model('weights/plate_detect.pt', device)  # 初始化检测模型
plate_rec_model = init_model(device, 'weights/plate_rec_color.pth', is_color=True)  # 初始化识别模型
