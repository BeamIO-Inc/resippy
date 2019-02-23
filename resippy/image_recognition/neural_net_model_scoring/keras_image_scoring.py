import numpy as np
import resippy.utils.image_utils.image_chipper as image_chipper
from keras.preprocessing.image import img_to_array
import scipy.misc as scipy_misc
from keras.models import Model


def chip_and_score_image(input_image,               # type: np.ndarray
                         trained_keras_model,       # type: Model
                         chip_size_x,               # type: int
                         chip_size_y,               # type: int
                         target_chip_size_x=224,    # type: int
                         target_chip_size_y=224,    # type: int
                         image_overlap_x_percent=0.50,  # type: float
                         image_overlap_y_percent=0.50,  # type: float
                         normalize_method="divide_by_255",  # type: str
                         labels=None,                       # type: list
                         thing_to_find=None,                # type: str
                         atk_chain_ledger=None,             # type: AlgorithmChain.ChainLedger
                         ):                                 # type: (...) -> (list, list)

    # remove the alpha channel if the image still has one
    if input_image.shape[2] == 4:
        input_image = input_image[:, :, 0: 3]

    npix_x_overlap = int(chip_size_x * image_overlap_x_percent)
    npix_y_overlap = int(chip_size_y * image_overlap_y_percent)
    image_chips, upper_left_yx_locs = image_chipper. \
        chip_entire_image_to_memory(input_image,
                                    chip_nx_pixels=chip_size_x, chip_ny_pixels=chip_size_y,
                                    npix_overlap_x=npix_x_overlap, npix_overlap_y=npix_y_overlap)
    scores = []

    n_chips = len(image_chips)
    for i, chip in enumerate(image_chips):
        chip = scipy_misc.imresize(chip, (target_chip_size_y, target_chip_size_x))
        chip = img_to_array(chip)  # shape is (ny, nx, 3)
        chip = np.expand_dims(chip, axis=0)  # Now shape is (1 ,ny, nx, 3)
        if normalize_method == "divide_by_255":
            chip = chip / 255.0
        elif normalize_method == "min_max_per_chip":
            chip = (chip - np.min(chip)) / (np.max(chip) - np.min(chip))

        preds = trained_keras_model.predict(chip)
        scores.append(preds)
        atk_chain_ledger.set_status('scoring image chips', (i / n_chips) * 100)

    scores = np.squeeze(np.array(scores))
    if labels is not None and thing_to_find is not None:
        thing_to_find_index = labels.index(thing_to_find)
        scores = scores[:, thing_to_find_index]
    return scores, upper_left_yx_locs
