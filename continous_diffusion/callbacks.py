import os
from composer import Callback, State, Logger, Algorithm
from IPython.display import clear_output
from .utils import median
from .diffusion import Diffusion

import wandb
from PIL import Image

class SchedulerUpdater(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch%self.frequency==0 and state.timestamp.batch!=0:
            curves_filename=f'training_figures/curves_{(self.model.n_parameters/1e6):.2f}M_{state.run_name}.png'
            datapoints_filename=f'training_figures/datapoints_{(self.model.n_parameters/1e6):.2f}M_{state.run_name}.png'

            self.model.noise_schedule.update_optimal_parameters()
            self.model.noise_schedule.plot_training_curves(
                f'CrossEntropy-Sigma Curve for {(self.model.n_parameters/1e6):.2f}M parameters every {self.frequency} batches, median={self.model.noise_schedule.medians[-1]}',
                filename=curves_filename
                )
            clear_output(wait=True)


            if wandb.run is not None: # for this to work you need the schedule update frequency to be a multiple of the plotting frequency
                file_paths=[curves_filename, datapoints_filename]
                # Ensure the files exist before opening them
                if all(os.path.exists(file_path) for file_path in file_paths):
                    images = [Image.open(file_path) for file_path in file_paths]
                    wandb.log({"training_images": [wandb.Image(image) for image in images]})
                else:
                    print(f"Files not found: {[file_path for file_path in file_paths if not os.path.exists(file_path)]}")



class PlottingData(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency

    def batch_start(self, state: State, logger: Logger):
        if state.timestamp.batch%self.frequency==0 and state.timestamp.batch!=0:
            self.model.noise_schedule.plot_entropy_time_curve(
                filename = f'training_figures/datapoints_{(self.model.n_parameters/1e6):.2f}M_{state.run_name}.png',
                title=f'entropy-time, median={self.model.noise_schedule.medians[-1]}'
                )


class WriteText(Callback):
    def __init__(self,frequency,model:Diffusion):
        super().__init__()
        self.model=model
        self.frequency=frequency
        self.table = wandb.Table(columns=["Generated Text"])
        self.generated_texts=[]

    def batch_start(self, state: State, logger: Logger):
        if state.timestamp.batch%self.frequency==0 and state.timestamp.batch!=0:
            filename=f'checkpoints/{(self.model.n_parameters/1e6):.2f}M_ep{state.timestamp.epoch}_ba{state.timestamp.batch}.txt'
            generated_text=self.model.generate_text(16,128,file=filename) # this is a list of strings with 16 elements

            if wandb.run is not None:
                self.generated_texts.append('\n'.join(generated_text))
                # Create a table with the accumulated generated texts
                table = wandb.Table(columns=["Generated Text"])
                for text in self.generated_texts:
                    table.add_data(text)

                wandb.log({"generated_text_table": table})

class LRMonitor(Callback):
    def __init__(self,plotting_frequency):
        super().__init__()
        self.plotting_frequency=plotting_frequency

    def batch_end(self, state: State, logger: Logger):
        if state.timestamp.batch%self.plotting_frequency==0 and state.timestamp.batch!=0:
            assert state.optimizers is not None, 'optimizers must be defined'
            for optimizer in state.optimizers:
                lrs = [group['lr'] for group in optimizer.param_groups]
                name = optimizer.__class__.__name__
                for idx, lr in enumerate(lrs):
                    print({f'lr-{name}/group{idx}': lr})


class FindUnused(Algorithm):
    #this class is needed to set find_unused_parameters to True when training with multi-gpu and using self-conditioning
    def match(self, event, state): return False
    def apply(event, state, logger): return None

    @property
    def find_unused_parameters(self): return True