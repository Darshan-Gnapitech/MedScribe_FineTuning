import os
import platform
import psutil
import torch
import subprocess

print("=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)

# OS
print(f"OS                : {platform.system()} {platform.release()}")
print(f"Python            : {platform.python_version()}")

# CPU
print(f"CPU               : {platform.processor()}")
print(f"Physical Cores    : {psutil.cpu_count(logical=False)}")
print(f"Logical Cores     : {psutil.cpu_count(logical=True)}")

# RAM
ram = psutil.virtual_memory()
print(f"RAM Total         : {ram.total / (1024**3):.2f} GB")
print(f"RAM Available     : {ram.available / (1024**3):.2f} GB")

# Disk
disk = psutil.disk_usage("/")
print(f"Disk Free         : {disk.free / (1024**3):.2f} GB")

print("\n" + "=" * 60)
print("CUDA / GPU INFORMATION")
print("=" * 60)

print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():

    print(f"CUDA Version      : {torch.version.cuda}")

    for i in range(torch.cuda.device_count()):

        props = torch.cuda.get_device_properties(i)

        print(f"\nGPU #{i}")
        print(f"Name              : {torch.cuda.get_device_name(i)}")
        print(f"VRAM Total        : {props.total_memory / (1024**3):.2f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Multiprocessors   : {props.multi_processor_count}")

        print(f"BF16 Support      : {torch.cuda.is_bf16_supported()}")

        try:
            print(
                f"Memory Allocated  : "
                f"{torch.cuda.memory_allocated(i)/(1024**2):.2f} MB"
            )

            print(
                f"Memory Reserved   : "
                f"{torch.cuda.memory_reserved(i)/(1024**2):.2f} MB"
            )
        except:
            pass

print("\n" + "=" * 60)
print("NVIDIA-SMI")
print("=" * 60)

try:
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader"
        ]
    ).decode()

    print(result)

except Exception as e:
    print("nvidia-smi not available:", e)

if torch.cuda.is_available():

    gpu = torch.cuda.get_device_name(0)

    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"GPU  : {gpu}")
    print(f"VRAM : {vram:.2f} GB")

    if vram < 6:
        print("Suitable: tiny/base only")

    elif vram < 8:
        print("Suitable: small + LoRA")

    elif vram < 12:
        print("Suitable: medium + LoRA")

    elif vram < 20:
        print("Suitable: large-v3 + LoRA")

    else:
        print("Suitable: full fine-tuning large models")
