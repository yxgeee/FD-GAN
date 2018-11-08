import os, sys
import os.path as osp
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.autograd import Variable

from reid import datasets
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomPairSampler
from reid.utils.data import transforms as T
from reid.utils.serialization import load_checkpoint
from reid.evaluators import CascadeEvaluator

from fdgan.options import Options
from fdgan.utils.visualizer import Visualizer
from fdgan.model import FDGANModel

def get_data(name, data_dir, height, width, batch_size, workers, pose_aug):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)

    # use combined trainval set for training as default
    train_loader = DataLoader(
        Preprocessor(dataset.trainval, root=dataset.images_dir, with_pose=True, pose_root=dataset.poses_dir,
                    pid_imgs=dataset.trainval_query, height=height, width=width, pose_aug=pose_aug),
        sampler=RandomPairSampler(dataset.trainval, neg_pos_ratio=3),
        batch_size=batch_size, num_workers=workers, pin_memory=False)

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, test_loader

def main():
    opt = Options().parse()
    dataset, train_loader, test_loader = get_data(opt.dataset, opt.dataroot, opt.height, opt.width, opt.batch_size, opt.workers, opt.pose_aug)

    dataset_size = len(dataset.trainval)*4
    print('#training images = %d' % dataset_size)

    model = FDGANModel(opt)
    visualizer = Visualizer(opt)

    evaluator = CascadeEvaluator(
                    torch.nn.DataParallel(model.net_E.module.base_model).cuda(),
                    model.net_E.module.embed_model,
                    embed_dist_fn=lambda x: F.softmax(Variable(x), dim=1).data[:, 0])
    if opt.stage!=1:
        print('Test with baseline model:')
        top1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, rerank_topk=100, dataset=opt.dataset)
        message = '\n Test with baseline model:  mAP: {:5.1%}  top1: {:5.1%}\n'.format(mAP, top1)
        visualizer.print_reid_results(message)

    total_steps = 0
    best_mAP = 0
    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        model.reset_model_status()

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if epoch % opt.save_step == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save(epoch)

        if epoch % opt.eval_step == 0 and opt.stage!=1:
            mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, top1=False)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            if is_best:
                model.save('best')
            message = '\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP, ' *' if is_best else '')
            visualizer.print_reid_results(message)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    # Final test
    if opt.stage!=1:
        print('Test with best model:')
        checkpoint = load_checkpoint(osp.join(opt.checkpoints, opt.name, '%s_net_%s.pth' % ('best', 'E')))
        model.net_E.load_state_dict(checkpoint)
        top1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, rerank_topk=100, dataset=opt.dataset)
        message = '\n Test with best model:  mAP: {:5.1%}  top1: {:5.1%}\n'.format(mAP, top1)
        visualizer.print_reid_results(message)

if __name__ == '__main__':
    main()
