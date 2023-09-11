import os
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
from utils import log_batch_to_wandb
import torchvision
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
import torchmetrics
import wandb
from torch.utils.data import DataLoader, ConcatDataset  # ,random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from utils import * 
import pandas as pd
from args import Args
from copy import deepcopy
from itertools import product
from collections import OrderedDict


# -- lightning datamodule
class Datamodule(pl.LightningDataModule):
    """Pytorch lightning Datamodule

    Args:
        pl (pytorch_lightning): parent module
    """

    def __init__(self, args):
        """Constructor.

        Args:
            args (Args): arguments parsed from Args.py which contains the configuration of the experiment.
        """
        super().__init__()
        self.args = args
        

    def setup(self, stage: str):
        
        if stage == "fit":
            
            self.train_data = Mydataset(mode=0, args=self.args)
            self.val_data = Mydataset(mode=1, args=self.args)

            if self.args.use_all_train_data:
                self.train_data = ConcatDataset(
                    [self.train_data, self.val_data])
                print('...Fitting Train+Validation sets')

            print('Size of train dataset : ', len(self.train_data))
            if not self.args.use_all_train_data:
                print('Size of validation dataset : ', len(self.val_data))
            print('Shape of data sample : ', self.val_data[0][0].shape)
        
        elif stage == 'test':
            self.test_data = Mydataset(mode=2, args=self.args)
            print('Size of test dataset : ', len(self.test_data))
        
        else:
            raise NotImplementedError


    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=self.args.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=self.args.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          pin_memory=self.args.pin_memory)

# Loss funcion from 
# ``Giunchiglia et al., Coherent Hierarchical Multi-Label Classification Networks``
class MCloss(torch.nn.Module):

    def __init__(self,args:Args) -> None:
        super().__init__()
        self.ancestryMatrix,_ = get_ancestry_matrix(args.label_hierarchy)
        self.ancestryMatrix = torch.Tensor(self.ancestryMatrix).unsqueeze(0)
        self.bce_withprobs = torch.nn.BCELoss(reduction='mean')
        self.loss_dice = smp.losses.DiceLoss(mode='multilabel', from_logits=False)
        # self.loss_focal = smp.losses.FocalLoss(mode=mode, gamma=args.focal_loss_gamma)
        self.add_dice = args.hierarchyloss_with_dice
        # self.use_focal = args.hierarchyloss_with_focal

    def forward(self,h:torch.Tensor,y:torch.Tensor):

        h = h.sigmoid()
        mcm = self.get_mcm_constraint(h,self.ancestryMatrix)
        h_ = h*y
        mcm_ = self.get_mcm_constraint(h_,self.ancestryMatrix)
        # get outputs & loss
        ###  mcm_*y := select only positve labels
        ### (1-y)*mcm select outputs for negative labels
        h_star = (1-y)*mcm + y*mcm_

        loss = 1.0*self.add_dice*self.loss_dice(h_star.double(), y.double()) +\
              self.bce_withprobs(h_star.double(),y.double()) 
        #*(1.0 - 1.0*self.use_focal)  +  (1.0*self.use_focal)*self.loss_focal(h_star.double(),y.double())
        
        return loss,mcm

    def get_mcm_constraint(self,x,R):
        """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R
        https://github.com/EGiunchiglia/C-HMCNN/blob/master/main.py
       """
        c_out = x.unsqueeze(1)
        c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
        R_batch = R.expand(len(x),R.shape[1], R.shape[1])
        final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
        return final_out.double()
        
# Loss funcion from 
# ``Li et al., Deep Hierarchical Semantic Segmentation``
class TreeMinLoss(torch.nn.Module):
    def __init__(self,args:Args) -> None:
        super().__init__()
        self.ancestryMatrix,_ = get_ancestry_matrix(args.label_hierarchy)
        self.ancestryMatrix = torch.Tensor(self.ancestryMatrix).unsqueeze(0)
        # self.indices_level1 ,self.indices_level2,self.indices_level3, self.indices_level4,self.indices_leaf = get_level_indices(args.label_hierarchy)
        self.bce_withprobs = torch.nn.BCELoss(reduction='mean')
        self.loss_dice = smp.losses.DiceLoss(mode='multilabel', from_logits=False)
        self.add_dice = args.hierarchyloss_with_dice

    def forward(self,h:torch.Tensor,y:torch.Tensor):

        #- get probablities
        h = h.sigmoid()
        # hierarchy consistent probabilities
        probs = self.get_probs(h)
        #-- get min on ancestors
        pv = self.get_min_constraint(h)
        #- get max on descendants
        mcm = self.get_max_constraint(h)

        # get outputs & loss
        ### pv*y := selects only positve labels
        ### (1-y)*mcm := selects outputs for negative labels
        h_star = (1-y)*mcm + pv*y 

        self.check_inputs(pv,y)
        self.check_inputs(mcm,y)
        self.check_inputs(h_star,y)
        loss = self.bce_withprobs(h_star.double(), y.double())
        loss = loss + self.loss_dice(h_star.double(), y.double()) if self.add_dice else loss

        return loss,probs

    def check_inputs(self,probs,labels):
        assert (labels.max() <= 1.0) and (labels.min() >= 0), f"Error in treeminloss : labels should be binary 0-1. Max={labels.max()}; Min={labels.min()}"
        assert (probs.max() <= 1.0) and (probs.min() >= 0), f"Error in treeminloss : probs should be binary 0-1. Max={probs.max()}; Min={probs.min()}"
    
    def get_max_constraint(self,x:torch.Tensor):
        """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R
        https://github.com/EGiunchiglia/C-HMCNN/blob/master/main.py
       """
        R = self.ancestryMatrix
        c_out = x.unsqueeze(1)
        c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
        R_batch = R.expand(len(x),R.shape[1], R.shape[1])
        final_out, indices = torch.max(R_batch*c_out.double(), dim = 2)
        return final_out.double()
    
    def get_min_constraint(self,x:torch.Tensor):
        """ Inspired by   https://github.com/EGiunchiglia/C-HMCNN/blob/master/main.py
       """
        R = self.ancestryMatrix.transpose(1,2)
        c_out = x.unsqueeze(1)
        c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
        R_batch = R.expand(len(x),R.shape[1], R.shape[1])
        selected = R_batch*c_out 
        selected = torch.where(selected == 0.0, 999.0,selected) # to enable min computation 
        final_out, indices = torch.min(selected, dim = 2)
        final_out = torch.where(final_out == 999.0, 0.0, final_out) # reverting torch.where operation
        return final_out.double()

    def get_probs(self,x:torch.Tensor):

        x = x.detach()
        R = self.ancestryMatrix
        R_ancestors = R.transpose(1,2).expand(len(x),R.shape[1], R.shape[1])
        c_out = x.unsqueeze(1)
        c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
        root_leaf_paths = (c_out*R_ancestors).sum(dim=2)
        _, indices = root_leaf_paths.max(dim=1)
        probs = R.transpose(1,2).squeeze()[indices]*x
        
        return probs

# Loss funcion from 
# ``Li et al., Deep Hierarchical Semantic Segmentation``
class TripletLoss(torch.nn.Module):

    def __init__(self, args:Args) -> None:
        super().__init__()

        self.args = args
        self.graph,_ = build_hierarchy_graph(args.label_hierarchy,
                                             nodeLabel_as_target=True)
        self.num_pairs = args.batch_size//2
        cosine = torch.nn.CosineSimilarity(dim=1)
        self.distance = lambda x,y : 0.5*(1.0 - cosine(x,y))
        self.device = 'cpu'
    
    def forward(self,x_embeddings:torch.Tensor,labels:torch.Tensor):

        positive_indices,negative_indices,is_valid,margins,flip_pos_neg = self.build_triplet(labels)
        num_samples = labels.shape[0]
        loss = 0
        num_valid = 0

        for i in range(num_samples):
            anchor   = x_embeddings.clone()[i,:].view(1,-1)
            positive = x_embeddings.clone()
            negative = x_embeddings[torch.from_numpy(negative_indices[i]).to(self.device),:].clone()
            margin_i = torch.Tensor(margins[i]).view(1,-1).to(self.device)

            is_valid_i = torch.Tensor(is_valid[i]).view(1,-1).to(self.device)
            flip_pos_neg_i = torch.Tensor(flip_pos_neg[i]).view(1,-1).to(self.device)

            loss_i = self.distance(anchor,positive) - self.distance(anchor,negative)
            loss_i = loss_i*torch.float_power(-1.0,flip_pos_neg_i) + margin_i  
            loss_i = torch.nn.functional.relu(loss_i*(is_valid_i+flip_pos_neg_i)) 

            loss = loss + loss_i.sum()
            num_valid = num_valid + (is_valid_i+flip_pos_neg_i).sum().item()

        return loss/num_valid
    
    def build_triplet(self,labels:torch.Tensor):

        num_samples = labels.shape[0]
        choices = np.arange(num_samples)
        positive_indices = [choices for _ in range(num_samples)]
        negative_indices = [np.random.permutation(choices) for _ in range(num_samples)]

        is_valid,margins,flip_pos_neg = self.select_valid_samples(self.graph,
                                                                  labels,
                                                                  positive_indices,
                                                                  negative_indices)
        
        return positive_indices,negative_indices,is_valid,margins,flip_pos_neg 
    
    def select_valid_samples(self,graph,labels,positive_indices,negative_indices):
        
        # indicator variables
        is_valid = list()
        flip_pos_neg = list()
        # margins
        margins = list()
        #- mapping function
        func = lambda x: tuple(x.to('cpu').numpy().flatten().tolist())
        D = 1./4 # hierarchy tree height inverse

        for label_index in range(labels.shape[0]):
            is_valid_i = list()
            margins_i = list()
            flip_pos_neg_i = list()
            for pos,neg in (zip(positive_indices[label_index],
                                negative_indices[label_index])):
                #- get tree distance
                d_positive = tree_distance(graph,source=labels[label_index],destination=labels[pos],mapping_func=func)            
                d_negative = tree_distance(graph,source=labels[label_index],destination=labels[neg],mapping_func=func)
                #- get margin
                margin = 0.1 + 0.25*abs(d_negative-d_positive)*D
                margins_i.append(margin)
                #- get indicator variables
                is_valid_i.append((d_negative > d_positive)*1.0)
                flip_pos_neg_i.append((d_negative < d_positive)*1.0)

            margins.append(margins_i)
            is_valid.append(is_valid_i)
            flip_pos_neg.append(flip_pos_neg_i)
            
        return  is_valid, margins, flip_pos_neg

# base model
class Neuralnetwork(torch.nn.Module):

    def __init__(self, args:Args) -> None:
        super().__init__()

        aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=None,               # dropout ratio, default is None
            activation=None,           # activation function, default is None
            classes=args.num_classes,  # presence of atrophy
            )
        self.args = args

        if self.args.encoder_name == 'efficientnetb3-pytorch':
            model = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
            model.classifier = torch.nn.Linear(in_features=1536,out_features=args.num_classes)
            self.encoder = model
            self.classification_head = None
            self.embeddings_size = 1536

        else:
            unet = smp.Unet(encoder_name=args.encoder_name,
                            encoder_depth=args.encoder_depth,
                            decoder_use_batchnorm=args.decoder_use_batchnorm,
                            decoder_channels=(256, 128, 64, 32, 16),
                            decoder_attention_type=args.decoder_attention_type,
                            encoder_weights=args.encoder_weights,
                            in_channels=args.in_channels,
                            classes=1,
                            activation=args.activation, 
                            aux_params=aux_params)
            self.encoder = deepcopy(unet.encoder)
            self.classification_head = deepcopy(unet.classification_head)
            self.embeddings_size = 384

            del unet # delete

        self.pool_and_flatten = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())

        # load weights if needed
        if os.path.exists(self.args.pretrained_encoder_weights):
            print(f'\nLoading encoder weights from: {self.args.pretrained_encoder_weights}\n')
            self.load_pretrained_weights()
    
    def forward(self,x):

        features = self.encoder(x)
        if self.args.encoder_name == 'efficientnetb3-pytorch':
            labels = features
            features = None
        else:
            labels = self.classification_head(features[-1])
            features = self.pool_and_flatten(features[-1])

        return labels,features

    def load_pretrained_weights(self,):

        path = Path(self.args.pretrained_encoder_weights)
        ckpts = torch.load(path,map_location='cpu')

        encoder_state = OrderedDict()

        for key,val in ckpts['state_dict'].items():
            if "model.encoder" in key:
                name = key.split('model.encoder.')[1]
                encoder_state[name] = val

        self.encoder.load_state_dict(encoder_state,strict=True)
        self.encoder.requires_grad_(not self.args.freeze_encoder)

# custom architectures for hierarchy
class CustomNetwork(torch.nn.Module):

    def __init__(self, args:Args) -> None:
        super().__init__()

        self.args = args
        model = Neuralnetwork(args=args)
        # if args.pre
        self.encoder = deepcopy(model.encoder)
        self.embeddings_size = model.embeddings_size
        self.device = 'cpu'  
        self.add_dummy_label = not self.args.customArch_useOtherLosses 

        # info on hierarchy
        self.all_labels = get_dataset_labels(args.label_hierarchy)
        self.indices_level1 ,self.indices_level2,self.indices_level3, self.indices_level4,self.indices_leaf = get_level_indices(args.label_hierarchy)
        self.level1_size = self.indices_level1.shape[0]
        self.level2_size = self.indices_level2.shape[0]
        self.level3_size = self.indices_level3.shape[0] + int(self.add_dummy_label)
        self.level4_size = self.indices_level4.shape[0] + int(self.add_dummy_label)

        # level logits
        self.x_1 = None
        self.x_2 = None
        self.x_3 = None
        self.x_4 = None
        
        self.pool_and_flatten = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())

        # activations
        if args.customArch_activation == 'RELU':
            self.activation = torch.nn.ReLU()
        elif args.customArch_activation == 'SELU':
            self.activation = torch.nn.SELU()
        elif args.customArch_activation == 'identity':
            self.activation = torch.nn.Identity()
        else:
            raise NotImplementedError

        # loss function
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        if args.criterion == "ce" :
            self.criterion = lambda ypred,ytruth: ce(ypred,torch.argmax(ytruth,dim=1).long())
        elif args.criterion == 'dice+ce':
            self.criterion = lambda ypred,ytruth: ce(ypred,torch.argmax(ytruth,dim=1).long()) +\
                                                  dice(ypred,torch.argmax(ytruth,dim=1).long())
        else:
            self.criterion = None
        
        # initilize Linear layers
        self.L1 = None
        self.L2 = None
        self.L3 = None
        self.L4 = None
        if args.customArch_strategy == 1:
            self.strategy_1()
        elif args.customArch_strategy == 2:
            self.strategy_2()
        elif args.customArch_strategy == 3:
            self.strategy_3()
        else:
            raise NotImplementedError

        del model

    def forward(self,images):

        x_embeddings = self.encoder(images)
        x_embeddings = self.pool_and_flatten(x_embeddings[-1])

        if self.args.customArch_strategy == 1:
            self.x_1 = self.activation(self.L1(x_embeddings))
            self.x_2 = self.activation(self.L2(self.x_1))
            self.x_3 = self.activation(self.L3(self.x_2))
            self.x_4 = self.L4(self.x_3)
        
        elif self.args.customArch_strategy == 2:
            self.x_1 = self.activation(self.L1(x_embeddings))
            self.x_2 = self.activation(self.L2(torch.cat([x_embeddings,self.x_1],dim=1)))
            self.x_3 = self.activation(self.L3(torch.cat([x_embeddings,self.x_2],dim=1)))
            self.x_4 = self.L4(torch.cat([x_embeddings,self.x_3],dim=1))
        
        elif self.args.customArch_strategy == 3:
            self.x_1 = self.activation(self.L1(x_embeddings))
            self.x_2 = self.activation(self.L2(torch.cat([x_embeddings,self.x_1],dim=1)))
            self.x_3 = self.activation(self.L3(torch.cat([x_embeddings,self.x_2,self.x_1],dim=1)))
            self.x_4 = self.L4(torch.cat([x_embeddings,self.x_3,self.x_2,self.x_1],dim=1))
        
        # get y_pred_logits
        y_pred_logits = torch.zeros((images.shape[0],self.all_labels.shape[0])).to(self.device)
        y_pred_logits[:,self.indices_level1] = self.x_1.clone()
        y_pred_logits[:,self.indices_level2] = self.x_2.clone()
        y_pred_logits[:,self.indices_level3] = self.x_3.clone()[:,int(self.add_dummy_label):]
        y_pred_logits[:,self.indices_level4] = self.x_4.clone()[:,int(self.add_dummy_label):]


        return y_pred_logits, x_embeddings
    
    def get_loss(self,labels:torch.Tensor):

        # get label per level
        x_1_truth = labels[:,self.indices_level1]
        x_2_truth = labels[:,self.indices_level2]
        x_3_truth = labels[:,self.indices_level3]
        x_4_truth = labels[:,self.indices_level4]

        if self.add_dummy_label:
            x_3_truth = torch.cat([(x_3_truth.sum(dim=1) < 1.0).float().view(-1,1),x_3_truth],dim=1)
            x_4_truth = torch.cat([(x_4_truth.sum(dim=1) < 1.0).float().view(-1,1),x_4_truth],dim=1)

        # compute loss
        loss = self.criterion(self.x_1, x_1_truth) +\
               self.criterion(self.x_2,x_2_truth) +\
               self.criterion(self.x_3,x_3_truth) +\
               self.criterion(self.x_4,x_4_truth)
        
        # mask_level3 = torch.where(x_3_truth.abs().sum(1) != 0.0, True,False)
        # mask_level4 = torch.where(x_4_truth.abs().sum(1) != 0.0, True,False)
        # if mask_level3.sum().item() > 0:
        #     loss = loss + self.criterion(self.x_3[mask_level3,:],
        #                                 x_3_truth[mask_level3,:]) 
        # if mask_level4.sum().item() > 0:
        #     loss = loss + self.criterion(self.x_4[mask_level4,:],
        #                              x_4_truth[mask_level4,:])

        return loss

    def get_probs(self,y_pred_logits:torch.Tensor):

        y_probs = torch.zeros_like(y_pred_logits)
        y_probs[:,self.indices_level1] = torch.softmax(self.x_1,dim=1)
        y_probs[:,self.indices_level2] = torch.softmax(self.x_2,dim=1)
        y_probs[:,self.indices_level3] = torch.softmax(self.x_3,dim=1)[:,int(self.add_dummy_label):]
        y_probs[:,self.indices_level4] = torch.softmax(self.x_4,dim=1)[:,int(self.add_dummy_label):]

        return y_probs
    
    def strategy_1(self) -> None:
        # La Grassa et al., Learn Class Hierarchy using Convolutional Neural Networks
        self.L1 = torch.nn.Linear(in_features=self.embeddings_size,out_features=self.level1_size)
        self.L2 = torch.nn.Linear(in_features=self.level1_size,out_features=self.level2_size)
        self.L3 = torch.nn.Linear(in_features=self.level2_size,out_features=self.level3_size)
        self.L4 = torch.nn.Linear(in_features=self.level3_size,out_features=self.level4_size)  

    def strategy_2(self) -> None:
        # Cerri et al., Reduction strategies for hierarchical multi-label classification in protein function prediction
        self.L1 = torch.nn.Linear(in_features=self.embeddings_size,
                                  out_features=self.level1_size)
        self.L2 = torch.nn.Linear(in_features=self.level1_size + self.embeddings_size,
                                  out_features=self.level2_size)
        self.L3 = torch.nn.Linear(in_features=self.level2_size + self.embeddings_size,
                                  out_features=self.level3_size)
        self.L4 = torch.nn.Linear(in_features=self.level3_size + self.embeddings_size,
                                  out_features=self.level4_size)
    
    def strategy_3(self,) -> None:
        # personal idea
        self.L1 = torch.nn.Linear(in_features=self.embeddings_size,
                                  out_features=self.level1_size)
        self.L2 = torch.nn.Linear(in_features=self.level1_size + self.embeddings_size,
                                  out_features=self.level2_size)
        self.L3 = torch.nn.Linear(in_features=self.level2_size + self.level1_size + self.embeddings_size,
                                  out_features=self.level3_size)
        self.L4 = torch.nn.Linear(in_features=self.level3_size + self.level2_size +  self.level1_size + self.embeddings_size,
                                  out_features=self.level4_size)

# -- Pytorch lightning model
class Architecture(pl.LightningModule):
    
    def __init__(self, args:Args):
        """Constructor.

        Args:
            args (Args): arguments parsed from Args.py which contains the configuration of the experiment.

        Raises:
            NotImplementedError: raised when args.criterion not in ["bce+focal", "dice", "bce", "jaccard", "lovasz","focal","dice+bce","dice+focal",'bce+mcc',"dice+bce+mcc",'dice+mcc']
        """
        super().__init__()

        self.training_step_outputs  = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # config
        self.args = args

        #-- Get all labels
        self.all_labels = get_dataset_labels(self.args.label_hierarchy)
        self.labels_names = pd.read_csv(self.args.label_names,sep=';').set_index('label',inplace=False)
        self.indices_level1 ,self.indices_level2,self.indices_level3, self.indices_level4,self.indices_leaf = get_level_indices(self.args.label_hierarchy)

        # run assertions
        assert self.args.model_name in ['default','customArch','baseline'],'One of these two should be given.'
        assert args.baseline_level in ['level1','level2','level3','level4','leaf'],'provide a correct input'
        self.levels_indices = dict(zip(['level1','level2','level3','level4','leaf'],
                                    [self.indices_level1 ,self.indices_level2,
                                    self.indices_level3, self.indices_level4,
                                    self.indices_leaf]))
        if self.args.model_name == 'baseline': 
            self.args.num_classes = self.levels_indices[self.args.baseline_level].shape[0]
        self.baseline_labels_indices = self.levels_indices[self.args.baseline_level]
        if self.args.criterion not in [
            "dice","bce",
            "focal","dice+bce",
            "mcloss","treeminloss",
            "treemin+triplet", "treetriplet",
            "ce","dice+ce"]:
            raise NotImplementedError
        
        # set threshold
        self.threshold_prediction = self.args.prediction_threshold

        #-- Metrics
        num_classes = self.baseline_labels_indices.shape[0] # number of classes on which to evaluate the models
        num_labels = self.args.num_labels # number of classes in hierarchy
        mode = 'multiclass' # metrics computed on the leafnode
        self.f1score_train_micro = torchmetrics.F1Score(task=mode,num_classes=num_classes,num_labels=num_labels,average='micro',threshold=self.threshold_prediction)
        self.f1score_valid_micro = torchmetrics.F1Score(task=mode,num_classes=num_classes,num_labels=num_labels,average='micro',threshold=self.threshold_prediction)
        self.f1score_test_micro = torchmetrics.F1Score(task=mode,num_classes=num_classes,num_labels=num_labels,average='micro',threshold=self.threshold_prediction)

        self.f1score_train_macro = torchmetrics.F1Score(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',threshold=self.threshold_prediction)
        self.f1score_valid_macro = torchmetrics.F1Score(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',threshold=self.threshold_prediction)
        self.f1score_test_macro = torchmetrics.F1Score(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',threshold=self.threshold_prediction)
        
        self.f1score_test_macro_level1 = torchmetrics.F1Score(task=mode,num_classes=self.indices_level1.shape[0],num_labels=num_labels,average='macro',threshold=self.threshold_prediction)
        self.f1score_test_macro_level2 = torchmetrics.F1Score(task=mode,num_classes=self.indices_level2.shape[0],num_labels=num_labels,average='macro',threshold=self.threshold_prediction)

        self.accuracy_train_micro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,num_labels=num_labels,average='micro',threshold=self.threshold_prediction)
        self.accuracy_valid_micro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,num_labels=num_labels,average='micro',threshold=self.threshold_prediction)
        self.accuracy_test_micro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,num_labels=num_labels,average='micro',threshold=self.threshold_prediction)

        self.accuracy_train_macro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',threshold=self.threshold_prediction)
        self.accuracy_valid_macro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',threshold=self.threshold_prediction)
        self.accuracy_test_macro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',threshold=self.threshold_prediction)

        self.aucpr_train_macro = torchmetrics.AveragePrecision(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',thresholds=5)
        self.aucpr_valid_macro = torchmetrics.AveragePrecision(task=mode,num_classes=num_classes,num_labels=num_labels,average='macro',thresholds=5)
        
        #mode2 = 'multiclass' #if self.args.model_name == 'baseline' else 'multilabel'
        # self.confmatrix_train = torchmetrics.StatScores(task=mode2,num_classes=num_classes,num_labels=num_labels,average=None,multidim_average='global',threshold=self.threshold_prediction)
        # self.confmatrix_valid = torchmetrics.StatScores(task=mode2,num_classes=num_classes,num_labels=num_labels,average=None,multidim_average='global',threshold=self.threshold_prediction)
        self.confmatrix_test = torchmetrics.StatScores(task=mode,num_classes=num_classes,num_labels=num_labels,average=None,multidim_average='global',threshold=self.threshold_prediction)

        #- these two are not used
        # self.aucpr_train_micro = torchmetrics.AveragePrecision(task='multilabel',num_labels=33,average='micro',thresholds=5)
        # self.aucpr_valid_micro = torchmetrics.AveragePrecision(task='multilabel',num_labels=33,average='micro',thresholds=5)

        self.metrics_objects = dict()
        self.metrics_objects['train'] = [['f1_score_macro',self.f1score_train_macro],
                                         ['f1_score_micro',self.f1score_train_micro],
                                         ['accuracy_macro',self.accuracy_train_macro],
                                         ['accuracy_micro',self.accuracy_train_micro],
                                         ['averagePrecision_macro',self.aucpr_train_macro],
                                        #  ['averagePrecision_micro',self.aucpr_train_micro],
                                         ]   
        self.metrics_objects['valid'] = [['f1_score_macro',self.f1score_valid_macro],
                                         ['f1_score_micro',self.f1score_valid_micro], 
                                         ['accuracy_macro',self.accuracy_valid_macro],
                                         ['accuracy_micro',self.accuracy_valid_micro],
                                         ['averagePrecision_macro',self.aucpr_valid_macro],
                                        #  ['averagePrecision_micro',self.aucpr_valid_micro],
                                         ]
        self.metrics_objects['test'] = [['f1_score_macro',self.f1score_test_macro],
                                         ['f1_score_micro',self.f1score_test_micro], 
                                         ['accuracy_macro',self.accuracy_test_macro],
                                         ['accuracy_micro',self.accuracy_test_micro],
                                         ]
        #-- Get model
        self.model = CustomNetwork(args=self.args)  if self.args.model_name == 'customArch' else Neuralnetwork(self.args)
        
        # -- declare loss functions
        mode = 'multiclass' if self.args.model_name == 'baseline' else 'multilabel'
        self.loss_dice = smp.losses.DiceLoss(mode=mode, from_logits=True)
        self.loss_focal = smp.losses.FocalLoss(mode=mode, gamma=self.args.focal_loss_gamma)
        self.loss_bce = smp.losses.SoftBCEWithLogitsLoss()
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='mean')
        # herarchy aware losses
        self.mcloss = MCloss(args=self.args) 
        self.treeminloss = TreeMinLoss(args=self.args)
        self.treeTripletloss = TripletLoss(args=self.args)
   
    def forward(self, image):

        # assign correct device
        if self.args.model_name != 'baseline':
            if self.device != self.mcloss.ancestryMatrix.device or self.device != self.treeminloss.ancestryMatrix.device:
                self.mcloss.ancestryMatrix = self.mcloss.ancestryMatrix.to(self.device)
                self.treeminloss.ancestryMatrix = self.treeminloss.ancestryMatrix.to(self.device)
            
            if self.treeTripletloss.device != self.device:
                self.treeTripletloss.device = self.device

            if self.args.model_name == 'customArch':
                if  self.model.device != self.device:
                    self.model.device = self.device 

        return self.model(image)

    def check_inputs(self,images,labels):
        assert images.ndim == 4
        h, w = images.shape[-2:]
        assert h % 32 == 0 and w % 32 == 0
        if self.args.model_name != 'baseline':
            assert labels.ndim == 2
            assert (labels.max() <= 1.0 or self.args.use_mixup) and labels.min(
            ) >= 0, "labels should be binary 0-1"
    
    def get_loss(self,logits_labels,labels,embeddings,stage):
        loss = 0
        if  self.args.model_name == 'customArch' and self.args.customArch_useOtherLosses == False:
            # compute loss as explained in publications
            loss = self.model.get_loss(labels.float())

        elif self.args.model_name == 'baseline':
            loss = self.loss_ce(logits_labels,labels)
            if self.args.criterion == 'dice+ce':
                loss = loss + self.loss_dice(logits_labels,labels)

        elif self.args.criterion == "dice":
            loss = self.loss_dice(logits_labels, labels)

        elif self.args.criterion == "focal":
            loss = self.loss_focal(logits_labels, labels)

        elif self.args.criterion == "bce":
            loss = self.loss_bce(logits_labels, labels)

        elif self.args.criterion == "dice+bce":
            loss = self.loss_bce(logits_labels, labels) + \
                self.loss_dice(logits_labels, labels)

        elif self.args.criterion == "mcloss":
            loss, mcm = self.mcloss(logits_labels,labels)
            #-- update prob_labels for compatibility with metrics logging
            prob_labels = mcm
            return loss,prob_labels
        
        elif self.args.criterion == "treeminloss":
            loss,prob_labels = self.treeminloss(logits_labels,labels)
            return loss,prob_labels

        elif self.args.criterion == "treemin+triplet":
            assert embeddings is not None, "Select a network that returns embeddings."
            beta = 0.5*np.cos(np.pi*0.5*self.current_epoch/(self.args.max_epochs-1))
            loss_2 = self.treeTripletloss(embeddings,labels)*beta 
            loss,prob_labels = self.treeminloss(logits_labels,labels)
            loss = loss+loss_2
            self.log(f'{stage}_Tree-Tripet-loss',
                    loss_2.item(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    logger=True)
            return loss,prob_labels
        
        elif self.args.criterion == "treetriplet":
            assert embeddings is not None, "Select a network that returns embeddings."
            loss = self.treeTripletloss(embeddings,labels) 
            
        else:
            raise NotImplementedError
        
        # assert not torch.isnan(loss), "NaN detected in loss"
        return loss        

    def shared_step(self, batch, stage, batch_idx):

        images, labels = batch
        labels = labels.float()
        # select labels for baseline
        if self.args.model_name == 'baseline' :
            labels = labels[:,self.baseline_labels_indices]
            labels = torch.argmax(labels,dim=1).long()

        # -- checks
        self.check_inputs(images,labels)

        # -- forward pass
        logits_labels, embeddings = self.forward(images)
        
        # -- get probabilities
        if self.args.model_name == 'baseline' : 
            prob_labels = torch.softmax(logits_labels.detach(),dim=1) # multiclass classification on leaf nodes
        elif self.args.model_name == 'customArch' and self.args.customArch_useOtherLosses == False:
            prob_labels = self.model.get_probs(logits_labels.detach()) # multiclass classification on each level of hierarchy
        else: 
            # multilabel classification
            prob_labels = logits_labels.detach().sigmoid()
        
        # -- compute loss
        if self.args.criterion in ['mcloss','treeminloss','treemin+triplet']:
            try:
                loss, prob_labels = self.get_loss(logits_labels,
                                                labels,
                                                embeddings,
                                                stage)
            except Exception as e:
                print(e,f"\nBatchIdx: {batch_idx}")
                assert 0>1,'Break..'
            
        else:
            loss = self.get_loss(logits_labels,
                                labels,
                                embeddings,
                                stage)

        # -- Log loss
        self.log(f"{stage}_loss",
            loss.item(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True)

        if self.args.criterion == 'treetriplet':
            # not needed
            return loss
        
        # select labels on which to compute evaluation metrics
        labels_leaf = torch.argmax(labels[:,self.baseline_labels_indices],dim=1).long() if self.args.model_name != 'baseline' else labels
        prob_labels_leaf = prob_labels[:,self.baseline_labels_indices].detach() if self.args.model_name != 'baseline' else prob_labels
        for name,metric in self.metrics_objects[stage]:                    
            metric.update(prob_labels_leaf, labels_leaf)
            self.log(f'{name}_{stage}',
                        metric,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        prog_bar=False)
        if stage == 'test':
            self.test_step_outputs.append([prob_labels_leaf.cpu().numpy(),labels_leaf.cpu().numpy()])
            self.confmatrix_test.update(prob_labels_leaf,labels_leaf)
            if self.args.model_name != 'baseline':
                self.f1score_test_macro_level1.update(prob_labels[:,self.indices_level1].detach(),
                                                      torch.argmax(labels[:,self.indices_level1],dim=1).long())
                self.f1score_test_macro_level2.update(prob_labels[:,self.indices_level2].detach(),
                                                      torch.argmax(labels[:,self.indices_level2],dim=1).long())
                self.log(f'{stage}_level1_f1_score',
                        self.f1score_test_macro_level1,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        prog_bar=False)
                self.log(f'{stage}_level2_f1_score',
                        self.f1score_test_macro_level2,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        prog_bar=False)
                
        return loss

    def shared_epoch_end(self, outputs, stage):

        if self.args.criterion == "treetriplet":
            # no need to compute metrics
            return None

        if stage == 'test':
            evaluation_labels = self.all_labels[self.baseline_labels_indices]
            evaluation_label_names = [self.labels_names.at[label,'name'] for label in evaluation_labels]

            # Log confusion matrix on leaf labels
            probs = np.vstack([prob for prob,label in outputs])
            labels = np.hstack([label for prob,label in outputs])
            wandb.log({f"conf_mat_{stage}":wandb.plot.confusion_matrix(y_true=labels,
                                                                        probs=probs,
                                                                       class_names=evaluation_label_names)})

            # log metris for each leaf label
            confmatrix = self.confmatrix_test.compute().cpu().numpy()
            self.confmatrix_test.reset()
            accuracy = lambda tp,fp,tn,fn : round((tp+tn)/(tp+fp+tn+fn),4)
            f1_score = lambda tp,fp,tn,fn : round(2*tp/(2*tp+fp+fn),4)
            try:
                data = dict()
                cols = ['label','accuracy','f1_score','num_examples','label_name']
                for ind,label in enumerate(evaluation_labels):
                    acc = accuracy(*confmatrix[ind,:-1].tolist())
                    f1 = f1_score(*confmatrix[ind,:-1].tolist())
                    num_examples = confmatrix[ind,-1]
                    label_name = self.labels_names.at[label,'name']
                    data[ind] = [label,acc,f1,num_examples,label_name]
                data = pd.DataFrame.from_dict(data, orient='index',columns=cols)
                wandb.log({f'PerClassMetrics_{stage}_epoch_{self.current_epoch}':wandb.Table(dataframe=data)})
                                
            except Exception as e:
                print('The metrics could not be computed -> ',e)

        return None

    def training_step(self, batch, batch_idx):        
        return self.shared_step(batch, "train", batch_idx)

    def on_training_epoch_end(self):
        out = self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

        return out

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid", batch_idx)

    def on_validation_epoch_end(self):
        out = self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return out

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test",batch_idx)
    
    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx):

        if isinstance(batch,tuple) or isinstance(batch,list):
            images,_ = batch
        else:
            images = batch
        logits_labels,embeddings = self.forward(images)
        
        #pred = (logits_labels.sigmoid() > self.threshold_prediction).float()

        return logits_labels,embeddings

    def configure_optimizers(self):

        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)

        if self.args.optimizer == "SGD":
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=0.9,
                nesterov=True)

        # --- Default
        interval = 'epoch'
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.args.lr_scheduler_mode,
            factor=self.args.lr_scheduler_factor,
            patience=self.args.lr_scheduler_patience,
            min_lr=self.args.min_lr)

        if self.args.lr_scheduler == "CosineAnnealingWarmRestarts":
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=self.args.T_0, T_mult=self.args.T_mult, eta_min=self.args.min_lr)
            
        elif self.args.lr_scheduler == "CosineAnnealingLR":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.args.max_epochs, eta_min=self.args.min_lr)

        elif self.args.lr_scheduler == "MultiplicativeLR":
            lr_func = lambda epoch: self.args.exp_lr_gamma*(epoch<=self.args.T_0) + 1.0*(epoch>self.args.T_0)
            sch = torch.optim.lr_scheduler.MultiplicativeLR(
                opt,
                lr_lambda= lr_func,
                last_epoch=-1)

        elif self.args.lr_scheduler == "OneCycleLR":
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=1e-3,
                total_steps=self.trainer.estimated_stepping_batches,
                anneal_strategy='cos')
            interval = 'step'
        else:
            print('using default scheduler: ReduceOnPlateau')
        out = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": self.args.metric_to_monitor_lr_scheduler,
                "frequency": 1,
                "interval": interval
            },
        }

        return out

# -- training routine
def train(args:Args):
    """training routine

    Args:
        args (Args): arguments parsed from Args.py which contains the configuration of the experiment.
    """
    
    pl.seed_everything(args.seed, workers=True)  # -- for reproducibility

    # -- Create model
    model = Architecture(args)
    data = Datamodule(args) 

    # -- Create trainer
    wandb_logger = WandbLogger(
        project=args.project_name,
        log_model=args.log_weights)
    print(args, "\n\n")

    # -- Lr monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpointer = pl.callbacks.ModelCheckpoint(
        monitor=args.metric_to_monitor_early_stop,
        mode=args.early_stop_mode,
        auto_insert_metric_name=True,
        save_on_train_epoch_end=False,
        save_weights_only=args.save_weights_only)
    callbacks = [lr_monitor, checkpointer]

    # -- Checking for early stopping
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=args.metric_to_monitor_early_stop,
                mode=args.early_stop_mode,
                check_on_train_epoch_end=False,
                patience=args.early_stopping_patience,
                verbose=False,
                min_delta=1e-4,
                stopping_threshold=0.99))

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         precision=args.precision,
                         max_time=args.max_time,
                         accelerator=args.device,
                         gradient_clip_val=10,
                         gradient_clip_algorithm='norm',
                         detect_anomaly=True,
                         max_steps=args.max_steps,
                         check_val_every_n_epoch=1,
                         logger=wandb_logger, 
                         log_every_n_steps=1,
                         callbacks=callbacks)
    # Train model
    trainer.fit(model, data)
    # Test model using best weights
    trainer.test(model=model,dataloaders=data,ckpt_path='best')

def test(args:Args):

    pl.seed_everything(args.seed, workers=True)  # -- for reproducibility

    # -- Create model
    model = Architecture(args)
    data = Datamodule(args) 
    # model = model.load_from_checkpoint(args.checkpoint_path,args=args)
    # print(type(model.model))
    # -- Create trainer
    wandb_logger = WandbLogger(
        project=args.project_name,
        log_model=args.log_weights)
    print(args, "\n\n")
    
    trainer = pl.Trainer(max_epochs=1,
                         precision=args.precision,
                         max_time=args.max_time,
                         accelerator=args.device,
                         detect_anomaly=True,
                         max_steps=-1,
                         check_val_every_n_epoch=1,
                         logger=wandb_logger, 
                         log_every_n_steps=1)

    trainer.test(model=model,
                 dataloaders=data,
                 ckpt_path=args.checkpoint_path)

