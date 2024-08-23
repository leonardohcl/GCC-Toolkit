from Keys import ConvNeuralNetwork
from TextUtils import ColorText
import tkinter as tk
from tkinter import filedialog
from Dataset import ImageDataset

class DataForm: 
    def __str__(self) -> str:
        txt = ''
        for field in self.__dict__.keys():
            txt += (f" - {field}: {self.__getattribute__(field)}\n")
        return txt
    
    @property
    def fields(self):
        return self.__dict__.keys()

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
    
    def query_values(self):
        pass

class CnnDataForm(DataForm):
    @property
    def available_models(self):
        return [ConvNeuralNetwork.RESNET_50, ConvNeuralNetwork.DENSENET_121, ConvNeuralNetwork.EFFICIENTNET_B2]

class TrainingParametersForm(DataForm):
    def __init__(self) -> None:
        self.learning_rate: float =  0.001
        self.learning_rate_drop:float = 0.75
        self.learning_rate_drop_frequency:int = 2
        self.number_of_folds: int = 10
        self.training_epochs:int = 10

    def query_values(self, skip_defaults:bool = False):
        if skip_defaults == False:
            is_default_training_params = self.promp_yes_or_no(F"\nWould you like to use the default parameters for training?\ndefault_parameters:\n{self}")
            if is_default_training_params: return
        
        print("\nPlease provide the parameters (if you want to keep the default please type nothing)")
        for field in self.fields:
            raw_value = input(f"{field}({self.__getattribute__(field)}) = ")
            if raw_value != '':
                value = int(raw_value) if isinstance(self.__getattribute__(field), int) else float(raw_value)
                self.__setattr__(field, value)

        is_correct = self.promp_yes_or_no(f"\nAre these values correct?\n{self}")
        if is_correct:
            return
        
        return self.query_values(True)

class ImageDatasetForm(DataForm):
    def __init__(self) -> None:
        self.csv_path:str = ''
        self.folder_path:str = ''
    
    def __str__(self) -> str:
        return f"Image folder: {ColorText.warning(self.folder_path)}\nDataset CSV: {ColorText.warning(self.csv_path)}"

    def _ask_dataset_csv_file(self):
        print("\nPlease select the .csv file with the input dataset information.")
        path = self.ask_file_path(allowed_extensions=[('csv files', ".csv")])
        print(f"Selected: {path}")
        classes = ImageDataset.get_csv_available_classes(path)
        print("Found the following classes in the image list:")
        for name in classes:
            print(f" - {name}")       
        self.csv_path = path

    def _ask_image_directory_path(self):
        print("\nPlease select the folder with the dataset images.")
        path = self.ask_dir_path()
        print(f"Selected folder: {path}")
        self.folder_path = path

    def query_values(self):
        self._ask_dataset_csv_file()
        self._ask_image_directory_path()

class TrainingForm(CnnDataForm):
    model_id: ConvNeuralNetwork

    def __init__(self) -> None:
        self.output_path: str = ''
        self.log_path:str = None
        self.will_log: bool = False
        self.is_transfer: bool = False
        self.is_freeze: bool = False
        self.training_parameters = TrainingParametersForm()
        self.dataset = ImageDatasetForm()

    def __str__(self) -> str:
        txt = ''
        txt += (f"Transfer Leraning: {self.is_transfer}\n")	
        if self.is_transfer: txt += (f"Freeze Layers: {self.is_freeze}\n")
        txt += f"{self.training_parameters}"
        txt += f"{self.dataset}\n"
        txt += ((f"Model saved at: {ColorText.info(self.output_path)}"))
        if self.will_log: txt += (f"Log saved at: {ColorText.info(self.log_path)}.log")
        return txt

    def query_values(self):
        model_idx = self.prompt_options("\nWhich model do you want to train?:", self.available_models)
        self.model_id = self.available_models[model_idx]
        self.is_transfer = self.promp_yes_or_no("\nWould you like to start with a trained model? (transfer learning)")
        if self.is_transfer:
            self.is_freeze = self.promp_yes_or_no("\nWould you like to train only the output of the model?")
        
        
        self.dataset.query_values()
        self.training_parameters.query_values()
        
        print("\nPlease select where and with which name should I save the training model")
        self.output_path = self.ask_save_path()
        print(f"Model will be saved as {self.output_path}")

        self.will_log = self.promp_yes_or_no("\nWould you like to save the log of the training?")
        if self.will_log:
            print("\nPlease select where and with which name should I save the training log")
            self.log_path = self.ask_save_path()
            print(f"Log will be saved as {self.output_path}")
        
class LayerWeightsExtractionForm(CnnDataForm):
    model_id: ConvNeuralNetwork

    def __init__(self) -> None:
        self.output_path: str = ''
        self.is_pre_trained = False
        self.is_transfer = False
        self.model_path = ''
        self.dataset = ImageDatasetForm()

    def __str__(self) -> str:
        txt = ''
        txt += (f"Pre-trained model: {ColorText.info(self.model_path if self.model_path else 'ImageNet' ) if self.is_pre_trained else 'no'}\n")	
        txt += f"{self.dataset}\n"
        txt += ((f"Output file: {ColorText.info(f"{self.output_path}.arff")}"))
        return txt

    def query_values(self):
        model_idx = self.prompt_options("\nWhich model do you want to train?:", self.available_models)
        self.model_id = self.available_models[model_idx]
        self.is_pre_trained = self.promp_yes_or_no("\nWould you like to use a trained model?")
        if self.is_pre_trained:
            self.is_transfer = self.promp_yes_or_no("\nDo you want to load the imagenet pretained model?")

            if self.is_transfer == False:
                print("\nPlease select the file contained your training model")
                self.model_path = self.ask_file_path()
                print(f"Selected: {self.model_path}")
        
        
        self.dataset.query_values()
        
        print("\nPlease select where and with which name should I save the resulting arff file")
        self.output_path = self.ask_save_path()
        print(f"File will be saved as {self.output_path}.arff")
     