
from MachineLearning import Trainer
from Forms import DataForm, TrainingForm
from TextUtils import ColorText

class UI:
    @classmethod
    def boot(self):
        print("\n\n-- GCC Toolkit v2.0 --")
        print("-- for more info, please visit https://github.com/leonardohcl/GCC-Toolkit  --")

        self.main_menu()

    @classmethod
    def main_menu(self):
        routine = DataForm.prompt_options("\n\nWhat would you like to do now?", (["Train a CNN", "exit"]))
        if routine == 0: self.cnn_training_routine()
        elif routine == 1: exit()

    @classmethod
    def cnn_training_routine(self):
        form = TrainingForm()
        form.query_values()
        Trainer.process_form(form)
        
        print(ColorText.success(f"\nFinished training model {form.model_id}"))
        print("Summary:")
        print(form)
        self.main_menu()