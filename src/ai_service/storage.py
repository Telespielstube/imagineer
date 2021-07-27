import torch, os

class Storage():

    def __init__(self, save_path, ai_service):
        self.path = save_path
        self.ai_service = ai_service

    # Saves the entire trained model to a specific path.
    def save_model(self, model):
        save_folder = '/home/marta/catkin_ws/src/imagineer/saved_models/'
        try:
            os.mkdir(save_folder)
        except FileExistsError:
            pass
        torch.save(self.ai_service.model, self.path)
        print('Model is saved')
    
    # Loads entire saved model.
    def load_model(self):
        self.ai_service.model = torch.load(self.path)