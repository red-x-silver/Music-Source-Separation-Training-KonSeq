from tempocnn.classifier import TempoClassifier
from tempocnn.feature import read_features
import glob
from tqdm.auto import tqdm
import re
import torch

model_name = 'fcn'
input_folder = ['D:/xiaowan/stage2/freesound-oneshots-dataset/Test_PriorSet']

# initialize the model (may be re-used for multiple files)
classifier = TempoClassifier(model_name)

# read the file's features
#features = read_features(input_file)

# estimate the global tempo
#tempo = classifier.estimate_tempo(features, interpolate=False)
#print(f"Estimated global tempo: {tempo}")

def get_mixture_paths(
    folder_path,
    extension = 'wav'
):
    """
    Retrieve paths to mixture files in the specified validation directories.

    Parameters:
    ----------
    valid_path : List[str]
        A list of directories to search for validation mixtures.

    extension : str
        File extension of the mixture files (e.g., 'wav').

    Returns:
    -------
    List[str]
        A list of file paths to the mixture files.
    """

    all_mixtures_path = []
    all_gt_tempo = []
    for path in folder_path:
        #part = sorted(glob.glob(f"{path}/*/mixture.{extension}"))
        part = sorted(glob.glob(f"{path}/*/sum_track_*{extension}"))
        if len(part) == 0:
            print(f'No validation wav found in: {path}')
        all_mixtures_path += part

        # Use regular expression to extract the number after 'bpm'
        for file in part:
            match = re.search(r'bpm(\d+)',file)

            # If the match is found, extract the number and convert it to an integer
            if match:
                bpm_value = int(match.group(1))  # Extract and convert to integer
                all_gt_tempo.append(bpm_value)
                print("Extracted BPM:", bpm_value)
            else:
                print("BPM not found in the file path")

    return all_mixtures_path, all_gt_tempo

def process_audio_files(
    input_folder,
    classifier,

    extension ='wav',
    verbose: bool = False,
    is_tqdm: bool = True
):
    """
    Process a list of audio files, perform source separation, and evaluate metrics.

    Parameters:
    ----------
    mixture_paths : List[str]
        List of file paths to the audio mixtures.
    model : torch.nn.Module
        The trained model used for source separation.
    args : Any
        Argument object containing user-specified options like metrics, model type, etc.
    config : Any
        Configuration object containing model and processing parameters.
    device : torch.device
        Device (CPU or CUDA) on which the model will be executed.
    verbose : bool, optional
        If True, prints detailed logs for each processed file. Default is False.
    is_tqdm : bool, optional
        If True, displays a progress bar for file processing. Default is True.

    Returns:
    -------
    Dict[str, Dict[str, List[float]]]
        A nested dictionary where the outer keys are metric names,
        the inner keys are instrument names, and the values are lists of metric scores.
    """
    mixture_paths, tempos = get_mixture_paths(input_folder)

    if is_tqdm:
        mixture_paths = tqdm(mixture_paths)

    avg_acc = 0
    for path, gt_tempo in zip(mixture_paths, tempos):
        #start_time = time.time()
       
        #folder = os.path.dirname(path)

        # read the file's features
        features = read_features(path)

        # estimate the global tempo
        tempo = classifier.estimate_tempo(features, interpolate=False)
        print(f"Estimated global tempo: {tempo}")

        # compute accuracy within +/- 5% of ground truth
        accuracy = torch.abs(tempo - torch.tensor(gt_tempo)) <= 0.05 * gt_tempo
        print(f'acc:{accuracy}')
        avg_acc += accuracy/len(mixture_paths)


       
    print(f'average acc:{avg_acc}') #for instr in ['kick','snare','hihats']:


    return avg_acc


avg_cc = process_audio_files(input_folder, classifier)