import platform
import sys

def main():
    print("=== System ===")
    print("OS:", platform.platform())
    print("Python:", sys.version)

    print("\n=== Ultralytics ===")
    try:
        import ultralytics
        print("ultralytics:", ultralytics.__version__)
    except Exception as e:
        print("Ultralytics error:", e)


        

    print("\n=== PyTorch ===")
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda version:", torch.version.cuda)
            print("gpu:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("PyTorch error:", e)

if __name__ == "__main__":
    main()
