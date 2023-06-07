from torchconvquality import measure_quality
import torch
from torchvision.models import resnet18
import csv
import os

def save_quality_metrics_csv(quality_metrics):
    print("saved")
    csv_file = f"Filteranalyse/quality_metrics.csv"
    write_header = not os.path.exists(csv_file) #Prüfe, ob Datei existiert
        
    with open(csv_file, mode="a", newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['Layer'] + list(quality_metrics.keys()))
        writer.writerow(list(quality_metrics.values()))



model = resnet18(num_classes =1000)
checkpoint = torch.load("output/20230605_115739_084677_cifar100_lowres_resnet18/checkpoints/best.ckpt")

#20230605_100006_504269_cifar100_lowres_resnet18
#20230605_115739_084677_cifar100_lowres_resnet18

model_state_dict = model.state_dict()
for name, param in checkpoint.items(): #Anpassung 'fc'-Schicht
    if name.startswith('fc'):
        model_state_dict[name] = param
        
model.load_state_dict(model_state_dict)                
#model.load_state_dict(checkpoint['state_dict'])                
quality_metrics = measure_quality(model)




for layer_name, metrics in quality_metrics.items():
    print(f"Layer: {layer_name}")
    print(f"Metrics: {metrics}")
    print("\n")

save_quality_metrics_csv(quality_metrics)
    
print("\nEnde")  


