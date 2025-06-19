from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from alg.modelopera import get_fea
from network import Adver_network, common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits


class Diversify(Algorithm):

    def __init__(self, args, class_weights=None):
        super(Diversify, self).__init__(args)
        self.featurizer = get_fea(args)
        self.dbottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)

        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)

        self.abottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck, args.layer)

        self.aclassifier = common_network.feat_classifier(
            args.num_classes*args.latent_domain_num, args.bottleneck, args.classifier)
        self.dclassifier = common_network.feat_classifier(
            args.latent_domain_num, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args
        self.class_weights = class_weights
        self.dropout = nn.Dropout(0.5)  # Add dropout

    def update_d(self, minibatch, opt):
        all_x1 = minibatch[0].cuda().float()
        all_d1 = minibatch[1].cuda().long()
        all_c1 = minibatch[4].cuda().long()
        z1 = self.dbottleneck(self.featurizer(all_x1))
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')
        
        # FIXED: Remove class weights for latent domain classification
        cd1 = self.dclassifier(z1)
        ent_loss = Entropylogits(cd1)*self.args.lam + \
            F.cross_entropy(cd1, all_c1)
                
        # Add L2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss = ent_loss + disc_loss + l2_lambda * l2_norm
        
        opt.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.cuda().float()
                index = data[-1]
                feas = self.dbottleneck(self.featurizer(inputs))
                outputs = self.dclassifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel')
        print(Counter(pred_label))
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def update(self, data, opt):
        all_x = data[0].cuda().float()
        all_y = data[1].cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))
        all_z = self.dropout(all_z)  # Apply dropout

        disc_input = all_z
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = data[4].cuda().long()

        disc_loss = F.cross_entropy(disc_out, disc_labels)
        
        # Keep class weights for activity classification
        all_preds = self.classifier(all_z)
        if self.class_weights is not None:
            classifier_loss = F.cross_entropy(all_preds, all_y, weight=self.class_weights)
        else:
            classifier_loss = F.cross_entropy(all_preds, all_y)
            
        # Add L2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss = classifier_loss + disc_loss + l2_lambda * l2_norm
        
        opt.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        opt.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].cuda().float()
        all_c = minibatches[1].cuda().long()
        all_d = minibatches[4].cuda().long()
        all_y = all_d*self.args.num_classes+all_c
        all_z = self.abottleneck(self.featurizer(all_x))
        all_z = self.dropout(all_z)  # Apply dropout
        
        all_preds = self.aclassifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        
        # Add L2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss = classifier_loss + l2_lambda * l2_norm
        
        opt.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        opt.step()
        return {'class': classifier_loss.item()}

    def predict(self, x, edge_index=None, batch_size=None):
        # Apply dropout only during training
        if self.training:
            z = self.dropout(self.bottleneck(self.featurizer(x)))
        else:
            z = self.bottleneck(self.featurizer(x))
        return self.classifier(z)

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))
