from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.lattice_structure import LatticeSDFStruct, constantLatvec
import splinepy
import torch


#*
# A model can be trained by using the train_deep_sdf function 
# that takes as input the experiment directory and the data directory.
import DeepSDFStruct.deep_sdf.training as training
training.train_deep_sdf("DeepSDFStruct/trained_models/test_experiment", data_dir)