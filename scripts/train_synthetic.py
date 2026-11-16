from argparse import ArgumentParser # Parse arguments and options typed in the command line after the script name (e.g., --n-epochs 5)
from typing import Tuple # Helper for type hints

# Each of the following 8 classes contains one group of parameters
from crystalsizer3d.args.dataset_training_args import DatasetTrainingArgs # DatasetTrainingArgs is a class in crystalsizer3d/args/dataset_training_args 
from crystalsizer3d.args.denoiser_args import DenoiserArgs
from crystalsizer3d.args.generator_args import GeneratorArgs
from crystalsizer3d.args.keypoint_detector_args import KeypointDetectorArgs
from crystalsizer3d.args.network_args import NetworkArgs
from crystalsizer3d.args.optimiser_args import OptimiserArgs
from crystalsizer3d.args.runtime_args import RuntimeArgs
from crystalsizer3d.args.transcoder_args import TranscoderArgs

from crystalsizer3d.nn.manager import Manager # Manager is the class that builds the networks and runs the training
from crystalsizer3d.util.utils import print_args # Helper that prints the parameters


def parse_arguments(printout: bool = True) -> Tuple[ # -> Tuple[...] means return 8 things of these 8 types in this order
    DatasetTrainingArgs,
    NetworkArgs,
    GeneratorArgs,
    DenoiserArgs,
    KeypointDetectorArgs,
    TranscoderArgs,
    OptimiserArgs,
    RuntimeArgs
]:
    """
    Parse command-line arguments and build parameter holders
    """
    parser = ArgumentParser(description='Train/Test crystal sizer network') # Create an empty ArgumentParser object: It's like a form with no fields

    # Add the fields: each setting class adds its options to the form (the parser)
    DatasetTrainingArgs.add_args(parser)
    NetworkArgs.add_args(parser)
    GeneratorArgs.add_args(parser)
    DenoiserArgs.add_args(parser)
    KeypointDetectorArgs.add_args(parser)
    TranscoderArgs.add_args(parser)
    OptimiserArgs.add_args(parser)
    RuntimeArgs.add_args(parser)

    # Do the parsing: read what has been typed in the command line after the script name, fill the defaults for what was not typed, and stores everything in the object args
    args = parser.parse_args()
    if printout:
        print_args(args)

    # Instantiate the 8 parameter holders: each takes only its own parameters from the object args
    dataset_args = DatasetTrainingArgs.from_args(args)
    net_args = NetworkArgs.from_args(args)
    generator_args = GeneratorArgs.from_args(args)
    denoiser_args = DenoiserArgs.from_args(args)
    keypoint_detector_args = KeypointDetectorArgs.from_args(args)
    transcoder_args = TranscoderArgs.from_args(args)
    optimiser_args = OptimiserArgs.from_args(args)
    runtime_args = RuntimeArgs.from_args(args)

    # Return these 8 parameter groups
    return dataset_args, net_args, generator_args, denoiser_args, keypoint_detector_args, transcoder_args, optimiser_args, runtime_args


def train():
    """
    Trains a network to estimate crystal growth parameters from images.
    """

    # Call parse_aeguments() and get 8 things back, which are assigned to 8 names
    dataset_args, net_args, generator_args, denoiser_args, keypoint_detector_args, transcoder_args, optimiser_args, runtime_args = parse_arguments()

    # Construct manager
    manager = Manager( # Create manager object and pass each parameter group by name
        dataset_args=dataset_args,
        network_args=net_args,
        generator_args=generator_args,
        denoiser_args=denoiser_args,
        keypoint_detector_args=keypoint_detector_args,
        transcoder_args=transcoder_args,
        optimiser_args=optimiser_args,
        runtime_args=runtime_args
    )

    # Generate the neural network computation graph (view in tensorboard)
    # manager.log_graph()

    # Starts the training
    manager.train(
        n_epochs=runtime_args.n_epochs # This is how a number typed in the command line reaches the training loop
    )


if __name__ == '__main__': # True only when this file is run directly (not when another file imports it). Then, train() is called
    train()
