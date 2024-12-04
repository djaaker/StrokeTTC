import torch

def convert_ckpt_to_pth(ckpt_path, pth_path):
    """
    Convert a PyTorch Lightning .ckpt file to a vanilla PyTorch .pth file.

    Args:
        ckpt_path (str): Path to the input .ckpt file.
        pth_path (str): Path to the output .pth file.
    """
    try:
        # Load the .ckpt file
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        # Extract the model's state dictionary
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise KeyError("The checkpoint does not contain a 'state_dict' key.")

        # Save the state dictionary as a .pth file
        torch.save(state_dict, pth_path)
        print(f"Converted {ckpt_path} to {pth_path} successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
if __name__ == "__main__":
    ckpt_file = "model.ckpt"  # Replace with your .ckpt file path
    pth_file = "model.pth"    # Replace with desired .pth file path
    convert_ckpt_to_pth(ckpt_file, pth_file)