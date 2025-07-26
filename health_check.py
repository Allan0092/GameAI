import torch

def main():
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.current_device())
    else:
        print("No GPU detected. Running on CPU.")

if __name__=="__main__":
    main()