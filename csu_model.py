"""
Store modular components for Jupyter Notebook
"""
import json
import numpy as np
import os
import csv
import logging
import random
import math
from sklearn import metrics
from scipy import stats
from os.path import join as pjoin

from collections import defaultdict
from itertools import combinations, izip
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torchtext import data
from util import MultiLabelField, ReversibleField, BCEWithLogitsLoss, MultiMarginHierarchyLoss


def get_ci(vals, return_range=False):
    if len(set(vals)) == 1:
        return (vals[0], vals[0])
    loc = np.mean(vals)
    scale = np.std(vals) / np.sqrt(len(vals))
    range_0, range_1 = stats.t.interval(0.95, len(vals) - 1, loc=loc, scale=scale)
    if return_range:
        return range_0, range_1
    else:
        return range_1 - loc


class Config(dict):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.__dict__.update(**kwargs)

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))


# then we can make special class for different types of model
# each config is used to build a classifier and a trainer, so one for each
class LSTMBaseConfig(Config):
    def __init__(self, emb_dim=100, hidden_size=512, depth=1, label_size=42, bidir=False,
                 c=False, m=False, dropout=0.2, emb_update=True, clip_grad=5., seed=1234,
                 rand_unk=True, run_name="default", emb_corpus="gigaword", avg_run_times=1,
                 conv_enc=False,
                 **kwargs):
        # run_name: the folder for the trainer
        super(LSTMBaseConfig, self).__init__(emb_dim=emb_dim,
                                             hidden_size=hidden_size,
                                             depth=depth,
                                             label_size=label_size,
                                             bidir=bidir,
                                             c=c,
                                             m=m,
                                             dropout=dropout,
                                             emb_update=emb_update,
                                             clip_grad=clip_grad,
                                             seed=seed,
                                             rand_unk=rand_unk,
                                             run_name=run_name,
                                             emb_corpus=emb_corpus,
                                             avg_run_times=avg_run_times,
                                             conv_enc=conv_enc,
                                             **kwargs)


class LSTM_w_C_Config(LSTMBaseConfig):
    def __init__(self, sigma_M, sigma_B, sigma_W, **kwargs):
        super(LSTM_w_C_Config, self).__init__(sigma_M=sigma_M,
                                              sigma_B=sigma_B,
                                              sigma_W=sigma_W,
                                              c=True,
                                              **kwargs)


class LSTM_w_M_Config(LSTMBaseConfig):
    def __init__(self, beta, **kwargs):
        super(LSTM_w_M_Config, self).__init__(beta=beta, m=True, **kwargs)


"""
Hierarchical ConvNet
"""


class ConvNetEncoder(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder, self).__init__()

        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']

        self.convnet1 = nn.Sequential(
            nn.Conv1d(self.word_emb_dim, 2 * self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convnet2 = nn.Sequential(
            nn.Conv1d(2 * self.enc_lstm_dim, 2 * self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convnet3 = nn.Sequential(
            nn.Conv1d(2 * self.enc_lstm_dim, 2 * self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convnet4 = nn.Sequential(
            nn.Conv1d(2 * self.enc_lstm_dim, 2 * self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0, 1).transpose(1, 2).contiguous()
        # batch, nhid, seqlen)

        sent = self.convnet1(sent)
        u1 = torch.max(sent, 2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb


"""
Normal ConvNet
"""
class NormalConvNetEncoder(nn.Module):
    def __init__(self, config):
        super(NormalConvNetEncoder, self).__init__()
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.enc_lstm_dim, kernel_size=(3, self.word_emb_dim), stride=(1, self.word_emb_dim))

    def encode(self, inputs):
        output = inputs.transpose(0, 1).unsqueeze(1) # [batch_size, in_kernel, seq_length, embed_dim]
        output = F.relu(self.conv(output)) # conv -> [batch_size, out_kernel, seq_length, 1]
        output = output.squeeze(3).max(2)[0] # max_pool -> [batch_size, out_kernel]
        return output

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: Variable(seqlen x batch x worddim)
        sent, sent_len = sent_tuple
        emb = self.encode(sent)
        return emb

"""
https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
352 stars
"""
class CNN_Text_Encoder(nn.Module):
    def __init__(self, config):
        super(CNN_Text_Encoder, self).__init__()

        self.word_emb_dim = config['word_emb_dim']

        # V = args.embed_num
        # D = args.embed_dim
        # C = args.class_num
        Ci = 1
        Co = config['kernel_num']  # 100
        Ks = config['kernel_sizes'] # '3,4,5'
        # len(Ks)*Co

        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, self.word_emb_dim)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        # self.dropout = nn.Dropout(args.dropout)
        # self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)

        x = x[0].transpose(0, 1).unsqueeze(1)
        # x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # x = self.dropout(x)  # (N, len(Ks)*Co)
        # logit = self.fc1(x)  # (N, C)
        return x



class Classifier(nn.Module):
    def __init__(self, vocab, config):
        super(Classifier, self).__init__()
        self.config = config
        self.drop = nn.Dropout(config.dropout)  # embedding dropout
        if config.conv_enc == 1:
            kernel_size = config.hidden_size / 8
            print(kernel_size)
            self.encoder = ConvNetEncoder({
                'word_emb_dim': config.emb_dim,
                'enc_lstm_dim': kernel_size if not config.bidir else kernel_size * 2
            })
            d_out = config.hidden_size if not config.bidir else config.hidden_size * 2
        elif config.conv_enc == 2:
            kernel_size = config.hidden_size
            print(kernel_size)
            self.encoder = NormalConvNetEncoder({
                'word_emb_dim': config.emb_dim,
                'enc_lstm_dim': kernel_size if not config.bidir else kernel_size * 2
            })
            d_out = config.hidden_size if not config.bidir else config.hidden_size * 2
        elif config.conv_enc == 3:
            kernel_num = config.hidden_size / 3
            kernel_num = kernel_num if not config.bidir else kernel_num * 2
            self.encoder = CNN_Text_Encoder({
                'word_emb_dim': config.emb_dim,
                'kernel_sizes': [3,4,5],
                'kernel_num': kernel_num
            })
            d_out = len([3,4,5]) * kernel_num
        else:
            self.encoder = nn.LSTM(
                config.emb_dim,
                config.hidden_size,
                config.depth,
                dropout=config.dropout,
                bidirectional=config.bidir)  # ha...not even bidirectional
            d_out = config.hidden_size if not config.bidir else config.hidden_size * 2

        self.out = nn.Linear(d_out, config.label_size)  # include bias, to prevent bias assignment
        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False

    def forward(self, input, lengths=None):
        output_vecs = self.get_vectors(input, lengths)
        return self.get_logits(output_vecs)

    def get_vectors(self, input, lengths=None):
        embed_input = self.embed(input)

        if self.config.conv_enc:
            output = self.encoder((embed_input, lengths.view(-1).tolist()))
            return output

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output)[0]

        # we ignored negative masking
        return output

    def get_logits(self, output_vec):
        if self.config.conv_enc:
            output = output_vec
        else:
            output = torch.max(output_vec, 0)[0].squeeze(0)
        return self.out(output)

    def get_softmax_weight(self):
        return self.out.weight


# this dataset can also take in 5-class classification
class Dataset(object):
    def __init__(self, path='./data/csu/',
                 dataset_prefix='snomed_multi_label_no_des_',
                 # test_data_name='adobe_abbr_matched_snomed_multi_label_no_des_test.tsv',
                 test_data_name='adobe_combined_abbr_matched_snomed_multi_label_no_des_test.tsv',
                 # change this to 'adobe_combined_abbr_matched_snomed_multi_label_no_des_test.tsv'
                 label_size=42, fix_length=None):
        self.TEXT = ReversibleField(sequential=True, include_lengths=True, lower=False, fix_length=fix_length)
        self.LABEL = MultiLabelField(sequential=True, use_vocab=False, label_size=label_size,
                                     tensor_type=torch.FloatTensor, fix_length=fix_length)

        # it's actually this step that will take 5 minutes
        self.train, self.val, self.test = data.TabularDataset.splits(
            path=path, train=dataset_prefix + 'train.tsv',
            validation=dataset_prefix + 'valid.tsv',
            test=dataset_prefix + 'test.tsv', format='tsv',
            fields=[('Text', self.TEXT), ('Description', self.LABEL)])

        self.external_test = data.TabularDataset(path=path + test_data_name,
                                                 format='tsv',
                                                 fields=[('Text', self.TEXT), ('Description', self.LABEL)])

        self.is_vocab_bulit = False
        self.iterators = []
        self.test_iterator = None

    def init_emb(self, vocab, init="randn", num_special_toks=2, silent=False):
        # we can try randn or glorot
        # mode="unk"|"all", all means initialize everything
        emb_vectors = vocab.vectors
        sweep_range = len(vocab)
        running_norm = 0.
        num_non_zero = 0
        total_words = 0
        for i in range(num_special_toks, sweep_range):
            if len(emb_vectors[i, :].nonzero()) == 0:
                # std = 0.5 is based on the norm of average GloVE word vectors
                if init == "randn":
                    torch.nn.init.normal(emb_vectors[i], mean=0, std=0.5)
            else:
                num_non_zero += 1
                running_norm += torch.norm(emb_vectors[i])
            total_words += 1
        if not silent:
            print("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
                running_norm / num_non_zero, num_non_zero, total_words))  # directly printing into Jupyter Notebook

    def build_vocab(self, config, silent=False):
        if config.emb_corpus == 'common_crawl':
            self.TEXT.build_vocab(self.train, vectors="glove.840B.300d")
            config.emb_dim = 300  # change the config emb dimension
        else:
            self.TEXT.build_vocab(self.train, vectors="glove.6B.{}d".format(config.emb_dim))
        self.is_vocab_bulit = True
        self.vocab = self.TEXT.vocab
        if config.rand_unk:
            if not silent:
                print("initializing random vocabulary")
            self.init_emb(self.vocab, silent=silent)

    def get_iterators(self, device, val_batch_size=128):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if len(self.iterators) > 0:
            return self.iterators  # return stored iterator

        # only get them after knowing the device (inside trainer or evaluator)
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (self.train, self.val, self.test), sort_key=lambda x: len(x.Text),  # no global sort, but within-batch-sort
            batch_sizes=(32, val_batch_size, val_batch_size), device=device,
            sort_within_batch=True, repeat=False)

        return train_iter, val_iter, test_iter

    def get_test_iterator(self, device):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet..needs to call build_vocab()")

        if self.test_iterator is not None:
            return self.test_iterator

        external_test_iter = data.Iterator(self.external_test, 128, sort_key=lambda x: len(x.Text),
                                           device=device, train=False, repeat=False, sort_within_batch=True)
        return external_test_iter

    def get_lm_iterator(self, device):
        # get language modeling data iterators
        pass


# compute loss
class ClusterLoss(nn.Module):
    def __init__(self, config, cluster_path='./data/csu/snomed_label_to_meta_grouping.json'):
        super(ClusterLoss, self).__init__()

        with open(cluster_path, 'rb') as f:
            label_grouping = json.load(f)

        self.meta_category_groups = label_grouping.values()
        self.config = config

    def forward(self, softmax_weight, batch_size):
        w_bar = softmax_weight.sum(1) / self.config.label_size  # w_bar

        omega_mean = softmax_weight.pow(2).sum()
        omega_between = 0.
        omega_within = 0.

        for c in xrange(len(self.meta_category_groups)):
            m_c = len(self.meta_category_groups[c])
            w_c_bar = softmax_weight[:, self.meta_category_groups[c]].sum(1) / m_c
            omega_between += m_c * (w_c_bar - w_bar).pow(2).sum()
            for i in self.meta_category_groups[c]:
                # this value will be 0 for singleton group
                omega_within += (softmax_weight[:, i] - w_c_bar).pow(2).sum()

        aux_loss = omega_mean * self.config.sigma_M + (omega_between * self.config.sigma_B +
                                                       omega_within * self.config.sigma_W) / batch_size

        return aux_loss


class MetaLoss(nn.Module):
    def __init__(self, config, cluster_path='./data/csu/snomed_label_to_meta_grouping.json',
                 label_to_meta_map_path='./data/csu/snomed_label_to_meta_map.json'):
        super(MetaLoss, self).__init__()

        with open(cluster_path, 'rb') as f:
            self.label_grouping = json.load(f)

        with open(label_to_meta_map_path, 'rb') as f:
            self.meta_label_mapping = json.load(f)

        self.meta_label_size = len(self.label_grouping)
        self.config = config

        # your original classifier did this wrong...found a bug
        self.bce_loss = nn.BCELoss()  # this takes in probability (after sigmoid)

    # now that this becomes somewhat independent...maybe you can examine this more closely?
    def generate_meta_y(self, indices, meta_label_size, batch_size):
        a = np.array([[0.] * meta_label_size for _ in range(batch_size)], dtype=np.float32)
        matched = defaultdict(set)
        for b, l in indices:
            if b not in matched:
                a[b, self.meta_label_mapping[str(l)]] = 1.
                matched[b].add(self.meta_label_mapping[str(l)])
            elif self.meta_label_mapping[str(l)] not in matched[b]:
                a[b, self.meta_label_mapping[str(l)]] = 1.
                matched[b].add(self.meta_label_mapping[str(l)])
        assert np.sum(a <= 1) == a.size
        return a

    def forward(self, logits, true_y, device):
        batch_size = logits.size(0)
        y_hat = torch.sigmoid(logits)
        meta_probs = []
        for i in range(self.meta_label_size):
            # 1 - (1 - p_1)(...)(1 - p_n)
            meta_prob = (1 - y_hat[:, self.label_grouping[str(i)]]).prod(1)
            meta_probs.append(meta_prob)  # in this version we don't do threshold....(originally we did)

        meta_probs = torch.stack(meta_probs, dim=1)
        assert meta_probs.size(1) == self.meta_label_size

        # generate meta-label
        y_indices = true_y.nonzero()
        meta_y = self.generate_meta_y(y_indices.data.cpu().numpy().tolist(), self.meta_label_size,
                                      batch_size)
        meta_y = Variable(torch.from_numpy(meta_y)) if device == -1 else Variable(torch.from_numpy(meta_y)).cuda(device)

        meta_loss = self.bce_loss(meta_probs, meta_y) * self.config.beta
        return meta_loss


# maybe we should evaluate inside this
# currently each Trainer is tied to one GPU, so we don't have to worry about
# Each trainer is associated with a config and classifier actually...so should be associated with a log
# Experiment class will create a central folder, and it will have sub-folder for each trainer
# central folder will have an overall summary...(Experiment will also have ways to do 5 random seed exp)
class Trainer(object):
    def __init__(self, classifier, dataset, config, save_path, device, load=False, run_order=0,
                 **kwargs):
        # save_path: where to save log and model
        if load:
            # or we can add a new keyword...
            if os.path.exists(pjoin(save_path, 'model-{}.pickle'.format(run_order))):
                self.classifier = torch.load(pjoin(save_path, 'model-{}.pickle'.format(run_order))).cuda(device)
            else:
                self.classifier = torch.load(pjoin(save_path, 'model.pickle')).cuda(device)
        else:
            self.classifier = classifier.cuda(device)

        self.dataset = dataset
        self.device = device
        self.config = config
        self.save_path = save_path

        self.train_iter, self.val_iter, self.test_iter = self.dataset.get_iterators(device)
        self.external_test_iter = self.dataset.get_test_iterator(device)

        if config.m:
            self.aux_loss = MetaLoss(config, **kwargs)
        elif config.c:
            self.aux_loss = ClusterLoss(config, **kwargs)

        self.bce_logit_loss = BCEWithLogitsLoss(reduce=False)

        need_grad = lambda x: x.requires_grad
        self.optimizer = optim.Adam(
            filter(need_grad, classifier.parameters()),
            lr=0.001)  # obviously we could use config to control this

        # setting up logging
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
        file_handler = logging.FileHandler("{0}/log.txt".format(save_path))
        self.logger = logging.getLogger(save_path.split('/')[-1])  # so that no model is sharing logger
        self.logger.addHandler(file_handler)

        self.logger.info(config)

    def load(self, run_order):
        self.classifier = torch.load(pjoin(self.save_path, 'model-{}.pickle').format(run_order)).cuda(self.device)

    def pretrain(self, epochs=15):
        # train loop
        # even though without attention...LM can still play a role here to learn word embeddings
        pass

    def train(self, run_order=0, epochs=5, no_print=True):
        # train loop
        exp_cost = None
        for e in range(epochs):
            self.classifier.train()
            for iter, data in enumerate(self.train_iter):
                self.classifier.zero_grad()
                (x, x_lengths), y = data.Text, data.Description

                # output_vec = self.classifier.get_vectors(x, x_lengths)  # this is just logit (before calling sigmoid)
                # final_rep = torch.max(output_vec, 0)[0].squeeze(0)
                # logits = self.classifier.get_logits(output_vec)

                logits = self.classifier(x, x_lengths)

                batch_size = x.size(0)

                if self.config.c:
                    softmax_weight = self.classifier.get_softmax_weight()
                    aux_loss = self.aux_loss(softmax_weight, batch_size)
                elif self.config.m:
                    aux_loss = self.aux_loss(logits, y, self.device)
                else:
                    aux_loss = 0.

                loss = self.bce_logit_loss(logits, y).mean() + aux_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.config.clip_grad)
                self.optimizer.step()

                if not exp_cost:
                    exp_cost = loss.data[0]
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

                if iter % 100 == 0:
                    self.logger.info(
                        "iter {} lr={} train_loss={} exp_cost={} \n".format(iter, self.optimizer.param_groups[0]['lr'],
                                                                            loss.data[0], exp_cost))
            self.logger.info("enter validation...")
            valid_em, micro_tup, macro_tup = self.evaluate(is_test=False)
            self.logger.info("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
                e + 1, self.optimizer.param_groups[0]['lr'], loss.data[0], valid_em
            ))

        # save model
        torch.save(self.classifier, pjoin(self.save_path, 'model-{}.pickle'.format(run_order)))

    def test(self, silent=False, return_by_label_stats=False, return_instances=False):
        self.logger.info("compute test set performance...")
        return self.evaluate(is_test=True, silent=silent, return_by_label_stats=return_by_label_stats,
                             return_instances=return_instances)

    def get_abstention_data_iter(self, data_iter):
        batched_x_list = []
        batched_y_list = []
        batched_y_hat_list = []
        batched_loss_list = []

        for iter, data in enumerate(data_iter):
            (x, x_lengths), y = data.Text, data.Description
            output_vec = self.classifier.get_vectors(x, x_lengths)  # this is just logit (before calling sigmoid)
            final_rep = torch.max(output_vec, 0)[0].squeeze(0)
            logits = self.classifier.get_logits(output_vec)
            loss = self.bce_logit_loss(logits, y)  # this per-example

            # We create new Tensor Variable
            batched_x_list.append(final_rep.detach())
            batched_y_list.append(y.detach())
            batched_y_hat_list.append(logits.detach())  # .tolist()
            batched_loss_list.append(loss.detach())  # .tolist()

        return batched_x_list, batched_y_list, batched_y_hat_list, batched_loss_list

    def get_abstention_data(self):
        # used by Abstention model
        self.classifier.eval()
        train_data = self.get_abstention_data_iter(self.train_iter)
        test_data = self.get_abstention_data_iter(self.test_iter)

        return train_data, test_data

    def save_error_examples(self, error_dict, label_names, save_address):
        import codecs
        # error_dict: {label: []}
        # creates 42 files in the given directory
        if not os.path.exists(pjoin(self.save_path, save_address)):
            os.makedirs(pjoin(self.save_path, save_address))

        for label_i, error_examples in error_dict.iteritems():
            file_name = label_names[label_i].replace('AND/OR', '').replace('and/or', '').replace('/', '')
            with codecs.open(pjoin(self.save_path, save_address, file_name + '.txt'), 'w', encoding='utf-8') as f:
                for e_tup in error_examples:
                    f.write(e_tup[0] + '\t' + '-'.join([str(x) for x in e_tup[1]]) + '\n')  # x tab y

    def get_error_examples(self, is_external=False, save_address=None, label_names=None):
        # this function is slower to run than evaluate()
        # save_address needs to point to a folder, not a file

        self.classifier.eval()
        data_iter = self.test_iter if not is_external else self.external_test_iter

        all_x, error_dict = [], defaultdict(list)
        all_preds, all_y_labels = [], []  # we traverse these two numpy array, then pick out things

        for iter, data in enumerate(data_iter):
            (x, x_lengths), y = data.Text, data.Description
            logits = self.classifier(x, x_lengths)
            preds = (torch.sigmoid(logits) > 0.5).data.cpu().numpy().astype(float)
            all_preds.append(preds)
            all_y_labels.append(y.data.cpu().numpy())

            orig_text = self.dataset.TEXT.reverse(x.data)
            all_x.extend(orig_text)

        preds = np.vstack(all_preds)
        ys = np.vstack(all_y_labels)

        for ith in range(len(all_x)):
            # traverse one by one to find ill match
            if (preds[ith] == ys[ith]).sum() == self.config.label_size:
                continue  # perfectly matched

            for jth in range(self.config.label_size):
                if preds[ith][jth] != ys[ith][jth]:
                    error_dict[jth].append((all_x[ith], ys[ith].nonzero()[0].tolist()))
                    # jth disease, append text, will result in duplication

        if save_address is not None:
            assert label_names is not None
            self.save_error_examples(error_dict, label_names, save_address)

        return error_dict

    def evaluate(self, is_test=False, is_external=False, silent=False, return_by_label_stats=False,
                 return_instances=False):
        self.classifier.eval()
        data_iter = self.test_iter if is_test else self.val_iter  # evaluate on CSU
        data_iter = self.external_test_iter if is_external else data_iter  # evaluate on adobe

        all_preds, all_y_labels = [], []

        for iter, data in enumerate(data_iter):
            (x, x_lengths), y = data.Text, data.Description
            logits = self.classifier(x, x_lengths)

            preds = (torch.sigmoid(logits) > 0.5).data.cpu().numpy().astype(float)
            all_preds.append(preds)
            all_y_labels.append(y.data.cpu().numpy())

        preds = np.vstack(all_preds)
        ys = np.vstack(all_y_labels)

        if not silent:
            self.logger.info("\n" + metrics.classification_report(ys, preds, digits=3))  # write to file

        # this is actually the accurate exact match
        em = metrics.accuracy_score(ys, preds)
        accu = np.array([metrics.accuracy_score(ys[:, i], preds[:, i]) for i in range(self.config.label_size)],
                        dtype='float32')
        p, r, f1, s = metrics.precision_recall_fscore_support(ys, preds, average=None)

        if return_by_label_stats:
            return p, r, f1, s, accu
        elif return_instances:
            return ys, preds

        micro_p, micro_r, micro_f1 = np.average(p, weights=s), np.average(r, weights=s), np.average(f1, weights=s)

        # compute Macro-F1 here
        # if is_external:
        #     # include clinical finding
        #     macro_p, macro_r, macro_f1 = np.average(p[14:]), np.average(r[14:]), np.average(f1[14:])
        # else:
        #     # anything > 10
        #     macro_p, macro_r, macro_f1 = np.average(np.take(p, [12] + range(21, 42))), \
        #                                  np.average(np.take(r, [12] + range(21, 42))), \
        #                                  np.average(np.take(f1, [12] + range(21, 42)))

        # we switch to non-zero macro computing, this can figure out boost from rarest labels
        if is_external:
            # include clinical finding
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), np.average(r[r.nonzero()]), \
                                         np.average(f1[f1.nonzero()])
        else:
            # anything > 10
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), \
                                         np.average(r[r.nonzero()]), \
                                         np.average(f1[f1.nonzero()])

        return em, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)


class AbstentionConfig(Config):
    def __init__(self, obj_loss=False, obj_accu=False,
                 inp_logit=False, inp_pred=False, inp_h=False, inp_conf=False, clip_grad=5., no_shrink=True,
                 dropout=0.):
        # some logic checks
        assert inp_logit + inp_pred + inp_h + inp_conf == 1, "only one input type"
        assert obj_loss + obj_accu == 1, "only one objective type"

        super(AbstentionConfig, self).__init__(obj_loss=obj_loss,
                                               obj_accu=obj_accu,  # objective is accu
                                               inp_logit=inp_logit,  # input is logit (before sigmoid)
                                               inp_pred=inp_pred,  # input is pred (after sigmoid)
                                               inp_h=inp_h,
                                               inp_conf=inp_conf,
                                               clip_grad=clip_grad,
                                               no_shrink=no_shrink,
                                               dropout=dropout)


class RejectModel(nn.Module):
    def __init__(self, config, deeptag_config):
        super(RejectModel, self).__init__()
        if config['inp_h']:
            reject_dim = deeptag_config.hidden_size
            if deeptag_config.bidir is True:
                reject_dim *= 2
        else:
            reject_dim = deeptag_config.label_size

        if config['no_shrink']:
            self.reject_model = nn.Sequential(
                nn.Linear(reject_dim, int(reject_dim)),
                nn.SELU(),
                nn.Linear(int(reject_dim), int(reject_dim)),
                nn.SELU(),
                nn.Linear(int(reject_dim), 1))
        else:
            self.reject_model = nn.Sequential(
                nn.Linear(reject_dim, int(reject_dim / 2.)),
                nn.SELU(),
                nn.Linear(int(reject_dim / 2.), int(reject_dim / 4.)),
                nn.SELU(),
                nn.Linear(int(reject_dim / 4.), 1))

    def pred(self, x):
        return self.reject_model(x)

    def reject(self, x, gamma=0.):
        # x: (batch_size, rej_dim)
        rej_choices = self.reject_model(x) > gamma
        return rej_choices


class Abstention(object):
    # Similar to Experiment class but used to train and manage Reject_Model
    def __init__(self, experiment, deeptag_config):
        self.mse_loss = torch.nn.MSELoss()
        self.sigmoid = torch.nn.Sigmoid()

        self.dataset = experiment.dataset
        self.experiment = experiment
        self.deeptag_config = deeptag_config

    def get_reject_model(self, config, gpu_id=-1):
        reject_model = RejectModel(config, self.deeptag_config)
        if gpu_id != -1:
            reject_model.cuda(gpu_id)
        return reject_model

    def train_loss(self, config, train_data, device, epochs=3, lr=0.001, print_log=False):
        # each training requires a new optimizer
        # these are already Variables
        # batched_x_list, batched_y_list, batched_y_hat_list, batched_loss_list = train_data
        # we used to return losses, but now we all know it works..so no need

        reject_model = self.get_reject_model(config, device)
        rej_optimizer = optim.Adam(reject_model.parameters(), lr=lr)

        exp_cost = None

        for n in range(epochs):
            iteration = 0
            # Already variables on CUDA devices
            print("training at epoch {}".format(n))
            for x, y, y_hat, orig_loss in izip(*train_data):
                reject_model.zero_grad()
                inp = None
                if config.inp_logit:
                    inp = y_hat
                elif config.inp_pred:
                    inp = torch.sigmoid(y_hat)
                elif config.inp_h:
                    inp = x
                elif config.inp_conf:
                    inp = torch.sigmoid(y_hat).cpu().apply_(lambda x: x if x >= 0.5 else 1 - x).cuda(device)

                pred_obj = torch.squeeze(reject_model.pred(inp))

                if config['obj_loss']:
                    true_obj = orig_loss.mean(dim=1)
                elif config['obj_accu']:
                    preds = torch.sigmoid(y_hat) > 0.5
                    true_obj = (preds.type_as(y) == y).type_as(y).mean(dim=1)

                loss = self.mse_loss(pred_obj, true_obj)

                loss.backward()
                torch.nn.utils.clip_grad_norm(reject_model.parameters(), config['clip_grad'])
                rej_optimizer.step()

                if not exp_cost:
                    exp_cost = loss.data[0]
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

                if iteration % 100 == 0 and print_log:
                    # avg_rej_rate = rej_scores.mean().data[0]
                    print("iter {} lr={} train_loss={} exp_cost={} \n".format(iteration,
                                                                              rej_optimizer.param_groups[0]['lr'],
                                                                              loss.data[0], exp_cost))
                iteration += 1

        return reject_model

    @staticmethod
    def compute_exactly_k(k, probs):
        # k: int
        # probs: [float] (confidence score!!)
        score = 0.
        for idx_tup in combinations(range(len(probs)), k):
            success = 1.
            for idx in idx_tup:
                success += math.log(probs[idx])
            failure = 1.
            for idx in set(range(len(probs))) - set(idx_tup):
                failure += math.log(1 - probs[idx])
            score += success + failure

        return score

    def drop(self, data_iter, reject_model, drop_portion, config, device, conf_abstention=False, return_dropped=False,
             weighted_f1=True):
        # apply to whatever documents we want and tag them with abstention priority scores
        # data_iter should be the test set of CSU
        # data_iter is actually not an iterator
        reject_model.eval()
        score_reverse = True if config['obj_loss'] and not conf_abstention else False

        prior_score_pred_y_pairs = []

        for x, y, y_hat, orig_loss in izip(*data_iter):
            if not conf_abstention:
                inp = None
                if config.inp_logit:
                    inp = y_hat
                elif config.inp_pred:
                    inp = torch.sigmoid(y_hat)
                elif config.inp_h:
                    inp = x
                elif config.inp_conf:
                    inp = torch.sigmoid(y_hat).cpu().apply_(lambda x: x if x >= 0.5 else 1 - x).cuda(device)

                pred_obj = torch.squeeze(reject_model.pred(inp))
                abs_scores = pred_obj.data.cpu().numpy().tolist()
            else:
                # conf_abstention methods
                abs_scores = []
                confs = torch.sigmoid(y_hat).cpu().apply_(lambda x: x if x >= 0.5 else 1 - x)
                confs = confs.data.numpy().tolist()
                for y_hhat in confs:
                    abs_score = self.compute_exactly_k(42, y_hhat)
                    abs_scores.append(abs_score)

            y_hat = y_hat.data.cpu().numpy()
            y = y.data.cpu().numpy()
            for i, abs_score in enumerate(abs_scores):
                preds = (y_hat[i] > 0.5).astype(float)
                y_np = y[i]
                prior_score_pred_y_pairs.append((abs_score, [preds, y_np]))

        # dropping process
        total_examples = len(prior_score_pred_y_pairs)
        drop_num = int(math.ceil(total_examples * drop_portion))

        # drop from smallest value to largest value (accuracy)
        sorted_list = sorted(prior_score_pred_y_pairs, key=lambda x: x[0], reverse=score_reverse)

        accepted_exs = sorted_list[drop_num:]  # take examples after drop_num
        rejected_exs = sorted_list[:drop_num]

        # then we compute the EM, micro-F1, macro-F1
        all_preds, all_y_labels = [], []
        for ex in accepted_exs:
            pred, y = ex[1]
            all_preds.append(pred);
            all_y_labels.append(y)

        if return_dropped:
            rej_preds = []
            rej_y_labels = []
            for ex in rejected_exs:
                pred, y = ex[1]
                rej_preds.append(pred);
                rej_y_labels.append(y)

        preds = np.vstack(all_preds)
        ys = np.vstack(all_y_labels)

        # this is actually the accurate exact match
        em = metrics.accuracy_score(ys, preds)
        p, r, f1, s = metrics.precision_recall_fscore_support(ys, preds, average=None)
        f1 = np.average(f1, weights=s) if weighted_f1 else np.average(f1[f1.nonzero()])

        if return_dropped:
            return rej_preds, rej_y_labels, em, f1

        return em, f1

    def get_ems_f1s(self, data_iter, model, config, device, conf_abstention=False, weighted_f1=True):
        # data_iter: test data
        # data_iter, reject_model, drop_portion, config, device
        ems = [];
        f1s = []
        rej_portions = np.linspace(0., 0.9, num=9)
        for rej_p in rej_portions:
            em, f1 = self.drop(data_iter, model, rej_p, config, device, conf_abstention, weighted_f1=weighted_f1)
            ems.append(em);
            f1s.append(f1)
        return ems, f1s

    def get_deeptag_data(self, run_order, device, rebuild_vocab=True):
        # send the model in here, we run it
        # need to specify which model to load (exact number)
        # the "data" obtained are universal -- meaning they stay the same during the
        # abstention module training
        if rebuild_vocab:
            self.dataset.build_vocab(self.deeptag_config, True)

        self.experiment.set_run_random_seed(run_order)

        trainer = self.experiment.get_trainer(self.deeptag_config, device, run_order, build_vocab=False, load=True)
        train_data, test_data = trainer.get_abstention_data()

        return train_data, test_data

    def save_deeptag_data(self, run_order, device, rebuild_vocab=True):
        # save it into the same format as LTR Vol 2.
        train_data, test_data = self.get_deeptag_data(run_order, device, rebuild_vocab)
        list_train_data = []
        list_test_data = []
        for i in range(len(train_data)):
            list_train_data.append([train_data[i][j].data.cpu().numpy().tolist() for j in range(len(train_data[i]))])
        for i in range(len(test_data)):
            list_test_data.append([test_data[i][j].data.cpu().numpy().tolist() for j in range(len(test_data[i]))])

        with open(pjoin(self.experiment.exp_save_path, self.deeptag_config.run_name,
                        "train_data.json"), 'wb') as f:
            json.dump(list_train_data, f)

        with open(pjoin(self.experiment.exp_save_path, self.deeptag_config.run_name,
                        "test_data.json"), 'wb') as f:
            json.dump(list_train_data, f)


# Experiment class can also be "handled" by Jupyter Notebook
# Usage guide:
# config also manages random seed. So it's possible to just swap in and out random seed from config
# to run an average, can write it into another function inside Experiment class called `repeat_execute()`
# also, currently once trainer is deleted, the classifier pointer would be lost...completely
class Experiment(object):
    def __init__(self, dataset, exp_save_path):
        """
        :param dataset: Dataset class
        :param exp_save_path: the overall saving folder
        """
        if not os.path.exists(exp_save_path):
            os.makedirs(exp_save_path)

        self.dataset = dataset
        self.exp_save_path = exp_save_path
        self.saved_random_states = [49537527, 50069528, 44150907, 25982144, 12302344]

        # we never want to overwrite this file
        if not os.path.exists(pjoin(exp_save_path, "all_runs_stats.csv")):
            with open(pjoin(self.exp_save_path, "all_runs_stats.csv"), 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['model', 'CSU EM', 'CSU micro-P', 'CSU micro-R', 'CSU micro-F1',
                                     'CSU macro-P', 'CSU macro-R', 'CSU macro-F1',
                                     'PP EM', 'PP micro-P', 'PP micro-R', 'PP micro-F1',
                                     'PP macro-P', 'PP macro-R', 'PP macro-F1'])

    def get_trainer(self, config, device, run_order=0, build_vocab=False, load=False, silent=True, **kwargs):
        # build each trainer and classifier by config; or reload classifier
        # **kwargs: additional commands for the two losses

        if build_vocab:
            self.dataset.build_vocab(config, silent)  # because we might try different word embedding size

        self.set_random_seed(config)

        classifier = Classifier(self.dataset.vocab, config)
        logging.info(classifier)
        trainer_folder = config.run_name if config.run_name != 'default' else self.config_to_string(config)
        trainer = Trainer(classifier, self.dataset, config,
                          save_path=pjoin(self.exp_save_path, trainer_folder),
                          device=device, load=load, run_order=run_order, **kwargs)

        return trainer

    def set_random_seed(self, config):
        seed = config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(config.seed)  # need to seed cuda too

    # I'm not sure if after setting random seed, should we set random state again...
    def set_run_random_seed(self, run_order):
        seed = self.saved_random_states[run_order]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def config_to_string(self, config):
        # we compare config to baseline config, if values are modified, we produce it into string
        model_name = "mod"  # this will be the "baseline"
        base_config = LSTMBaseConfig()
        for k, new_v in config.items():
            if k in base_config.keys():
                old_v = base_config[k]
                if old_v != new_v:
                    model_name += "_{}_{}".format(k, new_v)
            else:
                model_name += "_{}_{}".format(k, new_v)

        return model_name.replace('.', '').replace('-', '_')  # for 1e-3 to 1e_3

    def record_meta_result(self, meta_results, append, config, file_name='all_runs_stats.csv'):
        # this records result one line at a time!
        mode = 'a' if append else 'w'
        model_str = self.config_to_string(config)

        csu_em, csu_micro_tup, csu_macro_tup, \
        pp_em, pp_micro_tup, pp_macro_tup = meta_results

        with open(pjoin(self.exp_save_path, file_name), mode=mode) as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([model_str, csu_em, csu_micro_tup[0],
                                 csu_micro_tup[1], csu_micro_tup[2],
                                 csu_macro_tup[0], csu_macro_tup[1], csu_macro_tup[2],
                                 pp_em, pp_micro_tup[0], pp_micro_tup[1], pp_micro_tup[2],
                                 pp_macro_tup[0], pp_macro_tup[1], pp_macro_tup[2]])

    def record_per_run_result(self, meta_results, append, trainer_path, run_order, print_header=False):
        mode = 'a' if append else 'w'

        csu_em, csu_micro_tup, csu_macro_tup, \
        pp_em, pp_micro_tup, pp_macro_tup = meta_results

        with open(pjoin(trainer_path, "avg_run_stats.csv"), mode=mode) as f:
            csv_writer = csv.writer(f)
            if print_header:
                csv_writer.writerow(['run order', 'CSU EM', 'CSU micro-P', 'CSU micro-R', 'CSU micro-F1',
                                     'CSU macro-P', 'CSU macro-R', 'CSU macro-F1',
                                     'PP EM', 'PP micro-P', 'PP micro-R', 'PP micro-F1',
                                     'PP macro-P', 'PP macro-R', 'PP macro-F1'])

            csv_writer.writerow(['runtime_{}'.format(run_order), csu_em, csu_micro_tup[0],
                                 csu_micro_tup[1], csu_micro_tup[2],
                                 csu_macro_tup[0], csu_macro_tup[1], csu_macro_tup[2],
                                 pp_em, pp_micro_tup[0], pp_micro_tup[1], pp_micro_tup[2],
                                 pp_macro_tup[0], pp_macro_tup[1], pp_macro_tup[2]])

    def execute_trainer(self, trainer, train_epochs=5, append=True):
        # used jointly with `get_trainer()`
        # the benefit of this function is it will record meta-result into a file...
        # use this to "evaluate" a model
        trainer.train(epochs=train_epochs)
        csu_em, csu_micro_tup, csu_macro_tup = trainer.test()
        trainer.logger.info("===== Evaluating on PP data =====")
        pp_em, pp_micro_tup, pp_macro_tup = trainer.evaluate(is_external=True)
        trainer.logger.info("PP accuracy = {}".format(pp_em))
        self.record_meta_result([csu_em, csu_micro_tup, csu_macro_tup,
                                 pp_em, pp_micro_tup, pp_macro_tup],
                                append=append, config=trainer.config)

    def execute(self, config, device, train_epochs=5, append=True):
        # combined get_trainer() and execute_trainer()
        # this is also "training"...not evaluating
        agg_csu_ems, agg_pp_ems = [], []
        agg_csu_micro_tup, agg_csu_macro_tup = [], []
        agg_pp_micro_tup, agg_pp_macro_tup = [], []

        self.dataset.build_vocab(config, True)
        trainer_folder = config.run_name if config.run_name != 'default' else self.config_to_string(config)

        for run_order in range(config.avg_run_times):
            self.set_run_random_seed(run_order)  # hopefully this is enough...

            classifier = Classifier(self.dataset.vocab, config)
            trainer = Trainer(classifier, self.dataset, config,
                              save_path=pjoin(self.exp_save_path, trainer_folder),
                              device=device)

            trainer.train(run_order, train_epochs)
            csu_em, csu_micro_tup, csu_macro_tup = trainer.test()

            trainer.logger.info("===== Evaluating on PP data =====")
            pp_em, pp_micro_tup, pp_macro_tup = trainer.evaluate(is_external=True)
            trainer.logger.info("PP accuracy = {}".format(pp_em))

            print_header = run_order == 0

            self.record_per_run_result([csu_em, csu_micro_tup, csu_macro_tup,
                                        pp_em, pp_micro_tup, pp_macro_tup],
                                       append=append, trainer_path=trainer.save_path, run_order=run_order,
                                       print_header=print_header)

            agg_csu_ems.append(csu_em);
            agg_pp_ems.append(pp_em)
            agg_csu_micro_tup.append(np.array(csu_micro_tup));
            agg_csu_macro_tup.append(np.array(csu_macro_tup))
            agg_pp_micro_tup.append(np.array(pp_micro_tup));
            agg_pp_macro_tup.append(np.array(pp_macro_tup))

        csu_avg_em, pp_avg_em = np.average(agg_csu_ems), np.average(agg_pp_ems)
        csu_avg_micro, csu_avg_macro = np.average(agg_csu_micro_tup, axis=0).tolist(), np.average(agg_csu_macro_tup,
                                                                                                  axis=0).tolist()
        pp_avg_micro, pp_avg_macro = np.average(agg_pp_micro_tup, axis=0).tolist(), np.average(agg_pp_macro_tup,
                                                                                               axis=0).tolist()

        self.record_meta_result([csu_avg_em, csu_avg_micro, csu_avg_macro,
                                 pp_avg_em, pp_avg_micro, pp_avg_macro],
                                append=append, config=config)

    def delete_trainer(self, trainer):
        # move all parameters to cpu and then delete the pointer
        trainer.classifier.cpu()
        del trainer.classifier
        del trainer

    def compute_label_metrics_ci(self, config, list_metric_matrix):
        label_list_metric = [[] for _ in range(config.label_size)]
        mean, ubs, lbs = [], [], []

        for j in range(config.label_size):
            for mm in list_metric_matrix:
                label_list_metric[j].append(mm[j])

        for j in range(config.label_size):
            mean.append(np.mean(label_list_metric[j]))
            lb, ub = get_ci(label_list_metric[j], return_range=True)
            ubs.append(ub);
            lbs.append(lb)

        return mean, ubs, lbs

    def get_meta_result(self, config, device, rebuild_vocab=False, silent=False, return_avg=True,
                        print_to_file=False, file_name='', append=True):
        # returns: csu_avg_em, csu_avg_micro, csu_avg_macro, pp_avg_em, pp_avg_micro, pp_avg_macro
        # basically ONE row in the results table.
        # return_avg: return 5 runs individually (for std, ci calculation), or return average only
        if rebuild_vocab:
            self.dataset.build_vocab(config, True)

        agg_csu_ems, agg_pp_ems = [], []
        agg_csu_micro_tup, agg_csu_macro_tup = [], []
        agg_pp_micro_tup, agg_pp_macro_tup = [], []

        for run_order in range(config.avg_run_times):
            if not silent:
                print("Executing order {}".format(run_order))
            trainer = self.get_trainer(config, device, run_order, build_vocab=False, load=True)
            csu_em, csu_micro_tup, csu_macro_tup = trainer.test(silent=silent)
            pp_em, pp_micro_tup, pp_macro_tup = trainer.evaluate(is_external=True, silent=silent)

            agg_csu_ems.append(csu_em);
            agg_csu_micro_tup.append(np.array(csu_micro_tup))
            agg_csu_macro_tup.append(np.array(csu_macro_tup))
            agg_pp_micro_tup.append(np.array(pp_micro_tup));
            agg_pp_macro_tup.append(np.array(pp_macro_tup))
            agg_pp_ems.append(pp_em)

        csu_avg_em, pp_avg_em = np.average(agg_csu_ems), np.average(agg_pp_ems)
        csu_avg_micro, csu_avg_macro = np.average(agg_csu_micro_tup, axis=0).tolist(), np.average(agg_csu_macro_tup,
                                                                                                  axis=0).tolist()
        pp_avg_micro, pp_avg_macro = np.average(agg_pp_micro_tup, axis=0).tolist(), np.average(agg_pp_macro_tup,
                                                                                               axis=0).tolist()

        if print_to_file:
            assert file_name != ''
            self.record_meta_result([csu_avg_em, csu_avg_micro, csu_avg_macro,
                                     pp_avg_em, pp_avg_micro, pp_avg_macro],
                                    append=append, config=config, file_name=file_name)
        elif return_avg:
            return [csu_avg_em, csu_avg_micro[0],
                    csu_avg_micro[1], csu_avg_micro[2],
                    csu_avg_macro[0], csu_avg_macro[1], csu_avg_macro[2],
                    pp_avg_em, pp_avg_micro[0], pp_avg_micro[1], pp_avg_micro[2],
                    pp_avg_macro[0], pp_avg_macro[1], pp_avg_macro[2]]
        else:
            return [agg_csu_ems, agg_csu_micro_tup, agg_csu_macro_tup,
                    agg_pp_ems, agg_pp_micro_tup, agg_pp_macro_tup]

    def get_meta_header(self):
        # return a list of headers
        # in real scenario, the first column is often 'model name' or 'run order'
        return ['CSU EM', 'CSU micro-P', 'CSU micro-R', 'CSU micro-F1',
                'CSU macro-P', 'CSU macro-R', 'CSU macro-F1',
                'PP EM', 'PP micro-P', 'PP micro-R', 'PP micro-F1',
                'PP macro-P', 'PP macro-R', 'PP macro-F1']

    def evaluate(self, config, device, is_external=False, rebuild_vocab=False, silent=False,
                 return_f1_ci=False):
        # Similr to trainer.evaluate() signature
        # but allows to handle multi-run averaging!
        # we also always return by_label_stats
        # return: p,r,f1,s,accu

        if rebuild_vocab:
            self.dataset.build_vocab(config, True)

        agg_p, agg_r, agg_f1, agg_accu = 0., 0., 0., 0.
        agg_f1_list = []

        for run_order in range(config.avg_run_times):
            if not silent:
                print("Executing order {}".format(run_order))
            trainer = self.get_trainer(config, device, run_order, build_vocab=False, load=True)
            p, r, f1, s, accu = trainer.evaluate(is_test=True, is_external=is_external, return_by_label_stats=True,
                                                 silent=True)
            agg_p += p;
            agg_r += r;
            agg_f1 += f1;
            agg_accu += accu
            agg_f1_list.append(f1)

        if return_f1_ci:
            return self.compute_label_metrics_ci(config, agg_f1_list)

        agg_p, agg_r, agg_f1, agg_accu = agg_p / float(config.avg_run_times), agg_r / float(config.avg_run_times), \
                                         agg_f1 / float(config.avg_run_times), agg_accu / float(config.avg_run_times)

        return agg_p, agg_r, agg_f1, agg_accu

    def get_performance(self, config):
        # actually looks into trainer's actual file
        # returns: [(avg, std, CI), ...]
        trainer_folder = config.run_name if config.run_name != 'default' else self.config_to_string(config)

        stat_array = defaultdict(list)
        cat_size = 0.

        with open(pjoin(self.exp_save_path, trainer_folder, 'avg_run_stats.csv'), 'r') as f:
            csv_reader = csv.reader(f)
            for i, line in enumerate(csv_reader):
                if i == 0:
                    continue
                cat_size = len(line[1:])
                for j, stat in enumerate(line[1:]):
                    stat_array[j].append(float(stat))

        stats_res = [0.] * cat_size

        for j in range(cat_size):
            stats_res[j] = (np.mean(stat_array[j]), np.std(stat_array[j]), get_ci(stat_array[j]))

        return stats_res


# Important! Each time you use "get_iterators", must restore previous random state
# otherwise the sampling procedure will be different
def run_baseline(device):
    random.setstate(orig_state)
    lstm_base_c = LSTMBaseConfig(emb_corpus=emb_corpus, avg_run_times=avg_run_times,
                                 conv_enc=use_conv)
    curr_exp.execute(lstm_base_c, train_epochs=train_epochs, device=device)
    # trainer = curr_exp.get_trainer(config=lstm_base_c, device=device, build_vocab=True)
    # curr_exp.execute(trainer=trainer)


def run_bidir_baseline(device):
    random.setstate(orig_state)
    lstm_bidir_c = LSTMBaseConfig(bidir=True, emb_corpus=emb_corpus, avg_run_times=avg_run_times,
                                  conv_enc=use_conv)
    curr_exp.execute(lstm_bidir_c, train_epochs=train_epochs, device=device)
    # trainer = curr_exp.get_trainer(config=lstm_bidir_c, device=device, build_vocab=True)
    # curr_exp.execute(trainer=trainer)


def run_m_penalty(device, beta=1e-3, bidir=False):
    random.setstate(orig_state)
    config = LSTM_w_M_Config(beta, bidir=bidir, emb_corpus=emb_corpus, avg_run_times=avg_run_times,
                             conv_enc=use_conv)
    curr_exp.execute(config, train_epochs=train_epochs, device=device)
    # trainer = curr_exp.get_trainer(config=config, device=device, build_vocab=True)
    # curr_exp.execute(trainer=trainer)


def run_c_penalty(device, sigma_M, sigma_B, sigma_W, bidir=False):
    random.setstate(orig_state)
    config = LSTM_w_C_Config(sigma_M, sigma_B, sigma_W, bidir=bidir, emb_corpus=emb_corpus,
                             avg_run_times=avg_run_times, conv_enc=use_conv)
    curr_exp.execute(config, train_epochs=train_epochs, device=device)
    # trainer = curr_exp.get_trainer(config=config, device=device, build_vocab=True)
    # curr_exp.execute(trainer=trainer)

use_conv = 0

if __name__ == '__main__':
    # if we just call this file, it will set up an interactive console
    random.seed(1234)

    # we get the original random state, and simply reset during each run
    orig_state = random.getstate()

    action = raw_input("enter branches of default actions: active | baseline | meta | cluster \n")

    device_num = int(raw_input("enter the GPU device number \n"))
    assert -1 <= device_num <= 3, "GPU ID must be between -1 and 3"

    exp_name = raw_input("enter the experiment name, default is 'csu_new_exp', skip to use default: ")
    exp_name = 'csu_new_exp' if exp_name.strip() == '' else exp_name

    emb_corpus = raw_input("enter embedding choice, skip for default: gigaword | common_crawl \n")
    emb_corpus = 'gigaword' if emb_corpus.strip() == '' else emb_corpus
    assert emb_corpus == 'gigaword' or emb_corpus == 'common_crawl'

    avg_run_times = raw_input("enter run times (intger), maximum 5: \n")  # default 1, but should run 5 times
    avg_run_times = 1 if avg_run_times.strip() == '' else int(avg_run_times)
    avg_run_times = 5 if avg_run_times > 5 else avg_run_times

    dataset_number = raw_input("enter dataset name prefix id (1=snomed_multi_label_no_des_ \n "
                               "2=snomed_revised_fields_multi_label_no_des_ \n"
                               "3=snomed_all_fields_multi_label_no_des_): \n")

    if dataset_number.strip() == "":
        print("Default choice to 1")
        dataset_prefix = 'snomed_multi_label_no_des_'
    elif int(dataset_number) == 1:
        dataset_prefix = 'snomed_multi_label_no_des_'
    elif int(dataset_number) == 2:
        dataset_prefix = 'snomed_revised_fields_multi_label_no_des_'
    elif int(dataset_number) == 3:
        dataset_prefix = 'snomed_all_fields_multi_label_no_des_'

    conv_encoder = raw_input("Use conv_encoder or not? 0/1(Hierarchical)/2(Normal)/3(TextCNN) \n")
    assert (conv_encoder == '0' or conv_encoder == '1' or conv_encoder == '2' or conv_encoder == '3')

    global use_conv
    use_conv = int(conv_encoder.strip())

    train_epochs = raw_input("Enter the number of training epochs: (default 5) \n")
    if train_epochs.strip() == "":
        train_epochs = 5
    else:
        train_epochs = int(train_epochs.strip())

    print("loading in dataset...will take 3-4 minutes...")
    dataset = Dataset(dataset_prefix=dataset_prefix)

    curr_exp = Experiment(dataset=dataset, exp_save_path='./{}/'.format(exp_name))

    if action == 'active':
        import IPython;

        IPython.embed()
    elif action == 'baseline':
        # baseline LSTM
        run_baseline(device_num)
        run_bidir_baseline(device_num)
    elif action == 'meta':
        # baseline LSTM + M
        # run_m_penalty(device_num, beta=1e-3)
        # run_m_penalty(device_num, beta=1e-4)

        # run_baseline(device_num)
        # run_bidir_baseline(device_num)
        #
        # # baseline LSTM + M + bidir
        run_m_penalty(device_num, beta=1e-4, bidir=True)
        run_m_penalty(device_num, beta=1e-3, bidir=True)
        #
        # run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4, bidir=True)
        # run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3, bidir=True)
        #
        run_m_penalty(device_num, beta=1e-4)
        run_m_penalty(device_num, beta=1e-3)

        # run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4)
        # run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3)

    elif action == 'cluster':
        # baseline LSTM + C
        run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4)
        run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3)

        # baseline LSTM + C + bidir
        run_c_penalty(device_num, sigma_M=1e-5, sigma_B=1e-4, sigma_W=1e-4, bidir=True)
        run_c_penalty(device_num, sigma_M=1e-4, sigma_B=1e-3, sigma_W=1e-3, bidir=True)
    else:
        print("Non-identifiable action: {}".format(action))
