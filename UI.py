import tkinter as tk
from tkinter import filedialog
from Dataset import ImageDataset
import torch
import torch.nn as nn
from torchvision import models, transforms
from Dataset import ImageDataset
from MachineLearning import Trainer


KNOWN_MODELS = ["resnet50", "densenet121", "efficientnet_b2"]
DEFAULT_TRAINING_PARAMS:dict[str, float|int]  = {
    'learning_rate': 0.001,
    'learning_rate_drop': 0.75,
    'learning_rate_drop_frequency': 2,
    'number_of_folds': 10,
    'training_epochs': 10
}

PRETRAINED_MODEL_WEIGHTS = {
    'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
    'densenet121': models.DenseNet121_Weights.IMAGENET1K_V1,
    'efficientnet_b2': models.EfficientNet_B2_Weights.IMAGENET1K_V1
}

LOAD_MODEL_FUNCTIONS = {
    "resnet50": models.resnet50,
    "densenet121": models.densenet121,
    "efficientnet_b2": models.efficientnet_b2
}


class UI: 
    @classmethod
    def boot(self):
        print("\n\n-- GCC Toolkit v2.0 --")
        print("-- for more info, please visit https://github.com/leonardohcl/GCC-Toolkit  --")

        self.main_menu()

    @classmethod
    def main_menu(self):
        routine = self.prompt_options("\n\nWhat would you like to do now?", (["Train a CNN", "exit"]))
        if routine == 0: self.cnn_training_routine()
        elif routine == 1: exit()

    @classmethod
    def error_string(self, msg:str):
        return f"\033[91m{msg}\033[0m"

    @classmethod
    def warning_string(self, msg:str):
        return f"\033[93m{msg}\033[0m"

    @classmethod
    def success_string(self, msg:str):
        return f"\033[92m{msg}\033[0m"

    @classmethod
    def info_string(self, msg:str):
        return f"\033[96m{msg}\033[0m"

    @classmethod
    def prompt_options(self, query: str, opts:list[str], input_message:str = "_:") -> int:
        print(query)
        for idx in range(len(opts)):
            print(f"[{idx}] - {opts[idx]}")
        choice = int(input(input_message))
        if choice < 0 or choice >= len(opts):
            print(self.error_string("\ninvalid option!"))
            return self.prompt_options(query, opts)
        return choice
    
    @classmethod    
    def promp_yes_or_no(self, query:str) -> bool:
        choice = self.prompt_options(query, ["no", "yes"])
        return choice == 1

    @classmethod
    def ask_file_path(self, title:str = "", allowed_extensions:list[tuple] = []) -> str:
        root = tk.Tk()
        root.withdraw()
        return filedialog.askopenfilename(title=title, filetypes=allowed_extensions)

    @classmethod
    def ask_dir_path(self, title:str = "") -> str:
        root = tk.Tk()
        root.withdraw()
        return filedialog.askdirectory(title=title)

    @classmethod
    def ask_save_path(self, default_extension:str = None):
        root = tk.Tk()
        root.withdraw()
        return filedialog.asksaveasfilename(default_extension=default_extension)

    @classmethod 
    def ask_dataset_csv_file(self) -> str:
        print("\nPlease select the .csv file with the input dataset information.")
        csv_path = self.ask_file_path(allowed_extensions=[('csv files', ".csv")])
        print(f"Selected: {csv_path}")
        classes = ImageDataset.get_csv_available_classes(csv_path)
        print("Found the following classes in the image list:")
        for name in classes:
            print(f" - {name}")
        is_correct = self.promp_yes_or_no("\nIs this the right file?")
        if is_correct: return csv_path
        return self.ask_dataset_csv_file()

    @classmethod 
    def ask_image_directory_path(self) -> str:
        print("\nPlease select the folder with the dataset images.")
        folder_path = self.ask_dir_path()
        print(f"Selected folder: {folder_path}")
        return folder_path
        
    @classmethod 
    def ask_training_parameters(self, skip_defaults:bool = False) -> dict[str, float | int]:
        if skip_defaults == False:
            is_default_training_params = self.promp_yes_or_no(F"\nWould you like to use the default parameters for training?\ndefault_parameters:\n{DEFAULT_TRAINING_PARAMS}")
            if is_default_training_params:
                return DEFAULT_TRAINING_PARAMS.copy()
        
        training_parameters = {}
        print("\nPlease provide the parameters (if you want to keep the default please type nothing)")
        for key in DEFAULT_TRAINING_PARAMS.keys():
            training_parameters[key] = float(input(f"{key}({DEFAULT_TRAINING_PARAMS[key]}) = ") or DEFAULT_TRAINING_PARAMS[key])

        is_correct = self.promp_yes_or_no(f"\nAre these values correct?\n{training_parameters}")
        if is_correct: return training_parameters
        return self.ask_training_parameters(True)
        
    @classmethod
    def cnn_training_routine(self):
        model_idx = self.prompt_options("\nWhich model do you want to train?:", KNOWN_MODELS)
        model_id = KNOWN_MODELS[model_idx]
        is_transfer = self.promp_yes_or_no("\nWould you like to start with a trained model? (transfer learning)")
        is_freeze = False
        if is_transfer:
            is_freeze = self.promp_yes_or_no("\nWould you like to train only the output of the model?")
        csv_path = self.ask_dataset_csv_file()
        folder_path = self.ask_image_directory_path()
        training_parameters = self.ask_training_parameters()
        
        print("\nPlease select where and with which name should I save the training model")
        output_path = self.ask_save_path()

        will_log = self.promp_yes_or_no("\nWould you like to save the log of the training?")
        log_path = None
        if will_log:
            print("\nPlease select where and with which name should I save the training log")
            log_path = self.ask_save_path()


        class_list = ImageDataset.get_csv_available_classes(csv_path)

        # 1. Load model
        model = LOAD_MODEL_FUNCTIONS[KNOWN_MODELS[model_idx]](weights=PRETRAINED_MODEL_WEIGHTS[model_id] if is_transfer else None)

        # 2. Freeze the training for all layers
        # Obs. This step only applies for transfer learning
        if is_freeze:
            for param in model.parameters():
                param.requires_grad = False

        # 3. Update output to match number of classes
        Trainer.update_model_output_size(model, model_id, len(class_list))

        # 4. Create transforms for the data
        # Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
        # ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
        # using this values, but hey can be changed to values that best suits your case.
        transform_functions = [transforms.ToTensor()]
        if is_transfer:
            transform_functions.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        # 5. Create dataset. This type can be found in the file Dataset.py of this package
        # and gets the path to a csv with the list of the images file names and the base path to the folder of the
        # images. If you don't have the csv already, you can use the 'createFolderContentCsv' function
        # from the file FileHandling.py.
        dataset = ImageDataset(
            csv_path,
            folder_path,
            class_list,
            transform=transforms.Compose(transform_functions),
        )

        # 6. Call the training function
        print("\nTraining...")
        trained_model, learning_history = Trainer.k_fold_training(
            model,
            dataset,
            k = int(training_parameters['number_of_folds']),
            epochs = int(training_parameters['training_epochs']),
            learning_rate = training_parameters['learning_rate'],
            learning_rate_drop = training_parameters['learning_rate_drop'], 
            learning_rate_drop_step_size=training_parameters['learning_rate_drop_frequency'], 
            max_batch_size=5,
            plot_acc=False,
            plot_loss=False,
            log_filename=log_path,
            use_gpu=torch.cuda.is_available()
        )

        # 7. Save trained model (Optional)
        torch.save(trained_model.state_dict(), output_path)

        print(self.success_string(f"\nFinished training model {model_id}"))
        print("Summary:")
        print(f" - Transfer Leraning: {is_transfer}")	
        if is_transfer: print(f" - Freeze Layers: {is_freeze}")	
        for key in training_parameters:
            print(f" - {key}: {training_parameters[key]}")
        print(f"Image folder: {self.warning_string(folder_path)}")
        print(f"Dataset CSV: {self.warning_string(csv_path)}")
        print(f"Classes: {class_list}")
        print((f"Model saved at: {self.info_string(output_path)}"))
        if will_log: print(f"Log saved at: {self.info_string(log_path)}.log")
        self.main_menu()
