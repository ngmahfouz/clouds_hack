# Data

The clouds images should be contained in a single folder that we will reference as the variable `DATA_FOLDER_PATH`. It should contain 4 files:

- train.npy : containing the images in numpy format
- meto.npy : containing the meteorological variables associated with train.npy in numpy format
- files.npy : containing the names of the images in train.npy in numpy format
- metos_stats.npy : containing the means and standard deviations of each (8) meteorological variables


# Installation

## Using Singularity

All required packages are installed in the singularity image provided in ... To launch an interactive session with that image, run the following command:

    singularity shell --nv --bind "$DATA_FOLDER_PATH" $SINGULARITY_IMG_PATH

You may have to bind folders like ",/etc/pki/tls/certs/,/etc/pki/ca-trust/extracted/pem"

## Using pip

The following command will install the required packages listed in "requirements.txt"

    pip install -r requirements.txt

# Train a GAN 

The following command

	python infogan.py --o output_dir -c config_file.yaml

will train a GAN defined in "config_file.yaml" and output the associated files in "output_dir"

## Simple DCGAN

You can use the configuration file `dcgan_film01.yaml`

## InfoGAN

You can use the configuration file `default_training_config.yaml`

## FiLM (Feature wise Linear Modulation)

To use FiLM, you have to modify the field `film_layers` in the section `model` and specify the indices of the layers you want to FiLM separated by `a`.
For example, if you want to FiLM the first and second layers (corresponding to indices 0 and 1), you should have in your configuration file `film_layers: "0a1"`. An example can be found in `dcgan_film01.yaml`

## Other options

# Train a meteorological variables (metos) regressor

The following command

	python metos_regressor.py --o output_dir -c config_file.yaml -g gan_model_path.pth

will train a metos regressor defined in "config_file.yaml" and output the associated files in "output_dir". It will also evaluate the GAN (whose weights are `gan_model_path.pth`) on the test partition of the dataset. To only evaluate the latter GAN, you have to you the option `--train false`. If you want to predict the mean of the 8 metos rather than each of them individually, you can use the option `-n 1`
