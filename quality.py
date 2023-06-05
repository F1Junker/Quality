from torchconvquality import measure_quality
import torch
from torchvision.models import resnet18

model = resnet18(num_classes =1000)
checkpoint = torch.load("output/20230605_100006_504269_cifar100_lowres_resnet18/checkpoints/best.ckpt")

model_state_dict = model.state_dict()
for name, param in checkpoint.items():
    if name.startswith('fc'):
        model_state_dict[name] =param
        
model.load_state_dict(model_state_dict)                
#model.load_state_dict(checkpoint['state_dict'])                
quality_metrics = measure_quality(model)




for layer_name, metrics in quality_metrics.items():
    print(f"Layer: {layer_name}")
    print(f"Metrics: {metrics}")
    print("\n")
    
    
print("\nEnde")  
