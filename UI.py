import tkinter as tk
from tkinter import filedialog
from Dataset import ImageDataset
from Dataset import ImageDataset
from MachineLearning import Trainer
from Forms import TrainingForm, TrainingParametersForm, ImageDatasetForm

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
    def ask_training_parameters_form(self, skip_defaults:bool = False) -> TrainingParametersForm:
        form = TrainingParametersForm()
        if skip_defaults == False:
            is_default_training_params = self.promp_yes_or_no(F"\nWould you like to use the default parameters for training?\ndefault_parameters:\n{form}")
            if is_default_training_params:
                return form
        
        print("\nPlease provide the parameters (if you want to keep the default please type nothing)")
        for field in form.fields:
            raw_value = input(f"{field}({form.__getattribute__(field)}) = ")
            if raw_value != '':
                value = int(raw_value) if isinstance(form.__getattribute__(field), int) else float(raw_value)
                form.__setattr__(field, value)

        is_correct = self.promp_yes_or_no(f"\nAre these values correct?\n{form}")
        if is_correct: return form
        return self.ask_training_parameters_form(True)
    
    @classmethod
    def ask_dataset_form(self) -> ImageDatasetForm:
        form = ImageDatasetForm()
        form.csv_path = self.ask_dataset_csv_file()
        form.folder_path = self.ask_image_directory_path()
        return form

    @classmethod
    def ask_training_form(self) -> TrainingForm:
        form = TrainingForm()
        model_opts = Trainer.get_available_models()
        model_idx = self.prompt_options("\nWhich model do you want to train?:", [opt.value for opt in model_opts])
        form.model_id = model_opts[model_idx]
        form.is_transfer = self.promp_yes_or_no("\nWould you like to start with a trained model? (transfer learning)")
        if form.is_transfer:
            form.is_freeze = self.promp_yes_or_no("\nWould you like to train only the output of the model?")
        
        
        form.dataset = self.ask_dataset_form()
        form.training_parameters = self.ask_training_parameters_form()
        
        print("\nPlease select where and with which name should I save the training model")
        form.output_path = self.ask_save_path()
        print(f"Model will be saved as {form.output_path}")

        form.will_log = self.promp_yes_or_no("\nWould you like to save the log of the training?")
        if form.will_log:
            print("\nPlease select where and with which name should I save the training log")
            form.log_path = self.ask_save_path()
            print(f"Log will be saved as {form.output_path}")
        
        return form

    @classmethod
    def cnn_training_routine(self):
        form = self.ask_training_form()
        Trainer.process_form(form)
        
        print(self.success_string(f"\nFinished training model {form.model_id}"))
        print("Summary:")
        print(f" - Transfer Leraning: {form.is_transfer}")	
        if form.is_transfer: print(f" - Freeze Layers: {form.is_freeze}")	
        for field in form.training_parameters.fields:
            print(f" - {field}: {form.training_parameters.__getattribute__(field)}")
        print(f"Image folder: {self.warning_string(form.dataset.folder_path)}")
        print(f"Dataset CSV: {self.warning_string(form.dataset.csv_path)}")
        print((f"Model saved at: {self.info_string(form.output_path)}"))
        if form.will_log: print(f"Log saved at: {self.info_string(form.log_path)}.log")
        self.main_menu()