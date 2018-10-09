from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils import to_numpy
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True


def extract_embeddings(model, features, alpha, query=None, topk_gallery=None, rerank_topk=0, print_freq=500):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    pairwise_score = Variable(torch.zeros(len(query), rerank_topk, 2).cuda())
    probe_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    for i in range(len(query)):
        gallery_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in topk_gallery[i]], 0)
        pairwise_score[i, :, :] = model(Variable(probe_feature[i].view(1, -1).cuda(), volatile=True),
                                        Variable(gallery_feature.cuda(), volatile=True))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
         print('Extract Embedding: [{}/{}]\t'
               'Time {:.3f} ({:.3f})\t'
               'Data {:.3f} ({:.3f})\t'.format(
               i + 1, len(query),
               batch_time.val, batch_time.avg,
               data_time.val, data_time.avg))

    return pairwise_score.view(-1,2)


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), dataset=None, top1=True):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if top1:
      # Compute all kinds of CMC scores
      if not dataset:
        cmc_configs = {
            'allshots': dict(separate_camera_set=False,
                             single_gallery_shot=False,
                             first_match_break=False),
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False),
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}{:>12}{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                  .format(k, cmc_scores['allshots'][k - 1],
                          cmc_scores['cuhk03'][k - 1],
                          cmc_scores['market1501'][k - 1]))

        # Use the allshots cmc top-1 score for validation criterion
        return cmc_scores['allshots'][0]
      else:

        if (dataset == 'cuhk03'):
          cmc_configs = {
              'cuhk03': dict(separate_camera_set=True,
                                single_gallery_shot=True,
                                first_match_break=False),
              }
          cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                  query_cams, gallery_cams, **params)
                        for name, params in cmc_configs.items()}

          print('CMC Scores{:>12}'.format('cuhk03'))
          for k in cmc_topk:
              print('  top-{:<4}{:12.1%}'
                    .format(k,
                            cmc_scores['cuhk03'][k - 1]))
          # Use the allshots cmc top-1 score for validation criterion
          return cmc_scores['cuhk03'][0], mAP
        else:
          cmc_configs = {
              'market1501': dict(separate_camera_set=False,
                                 single_gallery_shot=False,
                                 first_match_break=True)
                      }
          cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                  query_cams, gallery_cams, **params)
                        for name, params in cmc_configs.items()}

          print('CMC Scores{:>12}'.format('market1501'))
          for k in cmc_topk:
              print('  top-{:<4}{:12.1%}'
                    .format(k,
                            cmc_scores['market1501'][k-1]))
          return cmc_scores['market1501'][0], mAP
    else:
      return mAP

class CascadeEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None):
        super(CascadeEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn

    def evaluate(self, data_loader, query, gallery, alpha=0, cache_file=None,
                 rerank_topk=75, second_stage=True, dataset=None, top1=True):
        # Extract features image by image
        features, _ = extract_features(self.base_model, data_loader)

        # Compute pairwise distance and evaluate for the first stage
        distmat = pairwise_distance(features, query, gallery)
        print("First stage evaluation:")
        if second_stage:
            evaluate_all(distmat, query=query, gallery=gallery, dataset=dataset, top1=top1)

            # Sort according to the first stage distance
            distmat = to_numpy(distmat)
            rank_indices = np.argsort(distmat, axis=1)

            # Build a data loader for topk predictions for each query
            topk_gallery = [[] for i in range(len(query))]
            for i, indices in enumerate(rank_indices):
                for j in indices[:rerank_topk]:
                    gallery_fname_id_pid = gallery[j]
                    topk_gallery[i].append(gallery_fname_id_pid)

            embeddings = extract_embeddings(self.embed_model, features, alpha,
                                    query=query, topk_gallery=topk_gallery, rerank_topk=rerank_topk)

            if self.embed_dist_fn is not None:
                embeddings = self.embed_dist_fn(embeddings.data)

            # Merge two-stage distances
            for k, embed in enumerate(embeddings):
                i, j = k // rerank_topk, k % rerank_topk
                distmat[i, rank_indices[i, j]] = embed
            for i, indices in enumerate(rank_indices):
                bar = max(distmat[i][indices[:rerank_topk]])
                gap = max(bar + 1. - distmat[i, indices[rerank_topk]], 0)
                if gap > 0:
                    distmat[i][indices[rerank_topk:]] += gap
            print("Second stage evaluation:")
        return evaluate_all(distmat, query, gallery, dataset=dataset, top1=top1)