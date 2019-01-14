from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras import optimizers
from keras.optimizers import Optimizer
from keras.callbacks import TensorBoard, ModelCheckpoint
from time import time
import tensorflow as tf
from tensorflow import Tensor
import os

# Do the following to allow our GPU memory to grow, otherwise we might run out of memory on the GPU
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


# TODO: add more optional paramters.  We usually like to improve performance by providing additional trainign data
# TODO rather than trying to manipulate / stretch / warp existing images in ways that may be unrealistic.
# Default rescale is 1.0/255, which assumes 8 bit imagery to be rescaled from 0-255 to 0-1
def setup_train_and_val_image_generators(
        image_size,                # type: int
        training_dir,              # type: str
        validation_dir,            # type: str
        batchsize=10,              # type: int
        rescale=1.0/255,           # type: float
        horizontal_flip=False,     # type: bool
        vertical_flip=False        # type: bool
        ):                         # type: (...) -> (DirectoryIterator, DirectoryIterator)

    train_datagen = image.ImageDataGenerator(rescale=rescale,
                                             horizontal_flip=horizontal_flip,
                                             vertical_flip=vertical_flip)

    training_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(image_size, image_size),
        batch_size=batchsize,
        class_mode='categorical')

    test_datagen = image.ImageDataGenerator(rescale=rescale)
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=batchsize,
        class_mode='categorical')

    return training_generator, validation_generator


# This is pretty simple, but if we don't specify where the weights are this will try to go grab the model
# from github.  Here we assume we've already downloaded the model and have put it in "weights_path"
def load_vgg16_model(weights_path,      # type: str
                     include_top=True,  # type: bool
                     ):                 # type:(...) -> Model
    vgg_pretrained_model = VGG16(weights=weights_path, include_top=include_top)
    return vgg_pretrained_model


def freeze_model_layers(model_to_freeze     # type: Model
                        ):                  # type: (...) -> None
    for layer in model_to_freeze.layers:
        layer.trainable = False


def unfreeze_model_layers(model_to_freeze       # type: Model
                        ):                      # type: (...) -> None
    for layer in model_to_freeze.layers:
        layer.trainable = True


# Note that we are using Dropout layer with value of 0.2 by default, i.e. we are discarding 20% weights
def create_final_dense_softmax_layer(input_model,               # type: Model
                                     n_classes,                 # type: int
                                     dropout=0.2,               # type: float
                                     dense_connections=128,     # type: int
                                     ):                         # type: (...) -> Tensor
    x = input_model.output
    x = Dense(dense_connections)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    return predictions


def train_model(base_model,                                 # type: Model
                training_generator,                         # type: training_generator
                validation_generator,                       # type: training_generator
                output_dir,                                 # type: str
                final_model_layers=None,                    # type: Tensor
                model_weights_file=None,                    # type: str
                model_output_fname="trained_model.h5",      # type: str
                n_epochs=20,                                # type: int
                loss="categorical_crossentropy",            # type: str
                optimizer=optimizers.Adam(),                # type: Optimizer
                metrics=["accuracy"]                        # type: list
                ):                                          # type: (...) -> None
    tensorboard_log_dir = os.path.join(output_dir, "logs")
    tensorboard_log_dir = os.path.join(tensorboard_log_dir, "{}".format(time()))
    tensorboard = TensorBoard(log_dir=tensorboard_log_dir)

    filepath = os.path.join(output_dir, model_output_fname)
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='min',
                                 period=1)

    callbacks_list = [checkpoint, tensorboard]
    if final_model_layers is None:
        model = base_model
    else:
        model = Model(inputs=base_model.input, outputs=final_model_layers)
    if model_weights_file is not None:
        model.load_weights(model_weights_file)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    num_training_img = len(training_generator.filenames)
    num_validation_img = len(validation_generator.filenames)

    steps_per_epoch = num_training_img / training_generator.batch_size
    validation_steps = num_validation_img/validation_generator.batch_size
    model.fit_generator(
            training_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            callbacks=callbacks_list,
            validation_data=validation_generator,
            validation_steps=validation_steps
            )
