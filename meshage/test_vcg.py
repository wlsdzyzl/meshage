from meshage.dataset import create_loader
from flemme.sampler import create_sampler
from flemme.trainer import test
from flemme.utils import load_config
from meshage.model_utils import create_model, test_run, save_data
def main():
    test_config = load_config()
    test(test_config,         
        create_model_fn = create_model,
        create_loader_fn = create_loader,
        create_sampler_fn = create_sampler,
        run_fn = test_run,
        save_data_fn = save_data)
    
