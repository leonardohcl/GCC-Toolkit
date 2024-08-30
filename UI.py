
from MachineLearning import Helper
from Forms import DataForm, TrainingForm, LayerWeightsExtractionForm, DatasetFolderMapForm
from TextUtils import ColorText
from File import Folder

class UI:
    @staticmethod
    def boot():
        print("\n\n-- GCC Toolkit v2.0 --")
        print("-- for more info, please visit https://github.com/leonardohcl/GCC-Toolkit  --")

        UI.main_menu()

    @staticmethod
    def main_menu():
        routines = [
            "Train a CNN", 
            "Extract CNN's last layer output", 
            "Create dataset CSV", 
            "exit"
        ]
        routine = DataForm.prompt_options("\n\nWhat would you like to do now?", routines)
        if routine == 0: UI.cnn_training_routine()
        elif routine == 1: UI.extract_model_output_routine()
        elif routine == 2: UI.map_dataset_folder_routine()
        else: exit()

    @staticmethod
    def cnn_training_routine():
        form = TrainingForm()
        form.query_values()
        Helper.train_cnn(form)
        
        print(ColorText.success(f"\nFinished training model {form.model_id}"))
        print("Summary:")
        print(form)
        UI.main_menu()

    @staticmethod
    def extract_model_output_routine():
        form = LayerWeightsExtractionForm()
        form.query_values()
        Helper.extract_output_layer_values(form)

        print(ColorText.success(f"\nFinished extracting data"))
        print("Summary:")
        print(form)
        UI.main_menu()

    @staticmethod
    def map_dataset_folder_routine():
        form = DatasetFolderMapForm()
        form.query_values()
        folder = Folder(form.folder_path)
        folder.generate_dataset_csv(form.output_path)
        UI.main_menu()