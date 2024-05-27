# Running the Program

**Note**: You will need Python 3.9.18 to run the code.

## Setup

Follow these steps to set up your environment and run the program:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

Depending on the dataset you are using, follow one of the instructions below:

### CIFAR 10 Dataset

Run the following command, replacing `<model_name>` with either `LiteResNet` or `resnet18`:

```bash
python main_cifar10.py --model <model_name>
```

### CIFAR 100 Dataset

Run the following command, also replacing `<model_name>` with either `LiteResNet` or `resnet18`:

```bash
python main_cifar100.py --model <model_name>
```

Make sure to replace `<model_name>` with your actual model name when running the commands.
