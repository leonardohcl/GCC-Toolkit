from Keys import ConvNeuralNetwork


class DataForm:
    
    def __str__(self) -> str:
        return str(self.__dict__) 
    
    @property
    def fields(self):
        return self.__dict__.keys()


class TrainingParametersForm(DataForm):
    def __init__(self) -> None:
        self.learning_rate: float =  0.001
        self.learning_rate_drop:float = 0.75
        self.learning_rate_drop_frequency:int = 2
        self.number_of_folds: int = 10
        self.training_epochs:int = 10


class ImageDatasetForm(DataForm):
    def __init__(self) -> None:
        self.csv_path:str = ''
        self.folder_path:str = ''


class TrainingForm(DataForm):
    model_id: ConvNeuralNetwork
    training_parameters: TrainingParametersForm
    dataset: ImageDatasetForm
    def __init__(self) -> None:
        self.output_path: str = ''
        self.log_path:str = None
        self.will_log: bool = False
        self.is_transfer: bool = False
        self.is_freeze: bool = False
