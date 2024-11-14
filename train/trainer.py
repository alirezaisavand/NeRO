import os
import random
from pathlib import Path

import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.name2dataset import name2dataset
from network.loss import name2loss
from network.renderer import name2renderer
from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import dummy_collate_fn
from NeRO.network.renderer import ConfigWrapper
from pointnerf.data import create_dataset

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]

def nearest_view(campos, raydir, xyz, id_list):

    cam_ind = torch.zeros([0, 1], device=campos.device, dtype=torch.long)
    step = 10000
    for i in range(0, len(xyz), step):
        dists = xyz[i:min(len(xyz), i + step), None, :] - campos[None, ...]  # N, M, 3
        dists_norm = torch.norm(dists, dim=-1)  # N, M
        dists_dir = dists / (dists_norm[..., None] + 1e-6)  # N, M, 3
        dists = dists_norm / 200 + (1.1 - torch.sum(dists_dir * raydir[None, :], dim=-1))  # N, M
        cam_ind = torch.cat([cam_ind, torch.argmin(dists, dim=1).view(-1, 1)], dim=0)  # N, 1
    return cam_ind

class Trainer:
    default_cfg = {
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg": {
            "lr_init": 2.0e-4,
            "lr_step": 100000,
            "lr_rate": 0.5,
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "novel_view_interval": 10000,
        "worker_num": 8,
        'random_seed': 6033,
    }

    def _init_dataset(self):
        self.train_set = name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)
        self.train_set = DataLoader(self.train_set, 1, True, num_workers=self.cfg['worker_num'],
                                    collate_fn=dummy_collate_fn)
        print(f'train set len {len(self.train_set)}')
        self.val_set_list, self.val_set_names = [], []
        dataset_dir = self.cfg['dataset_dir']
        for val_set_cfg in self.cfg['val_set_list']:
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            val_set = name2dataset[val_type](val_cfg, False, dataset_dir=dataset_dir)
            val_set = DataLoader(val_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
            self.val_set_list.append(val_set)
            self.val_set_names.append(name)
            print(f'{name} val set len {len(val_set)}')

    def _init_network(self):
        self.network = name2renderer[self.cfg['network']](self.cfg).cuda()

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training for NeuRay
        if self.cfg['multi_gpus']:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network = self.network
            self.train_losses = self.val_losses

        if self.cfg['optimizer_type'] == 'adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type'] == 'sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator = ValidationEvaluator(self.cfg)
        self.lr_manager = name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
        self.optimizer = self.lr_manager.construct_optimizer(self.optimizer, self.network)


    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}

        # Here we set options
        self.opt = ConfigWrapper(self.cfg)

        torch.manual_seed(self.cfg['random_seed'])
        np.random.seed(self.cfg['random_seed'])
        random.seed(self.cfg['random_seed'])
        self.model_name = cfg['name']
        self.model_dir = os.path.join('data/model', cfg['name'])
        if not os.path.exists(self.model_dir): Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        self.pth_fn = os.path.join(self.model_dir, 'model.pth')
        self.best_pth_fn = os.path.join(self.model_dir, 'model_best.pth')

    # Here we use a function from pointnerf train_ft.py


    # Here we load initial points embeddings
    def load_init_points(self):

        opt = self.opt
        train_dataset = create_dataset(opt)
        normRw2c = train_dataset.norm_w2c[:3, :3]
        points_xyz_all = None

        with torch.no_grad():
            # if len([n for n in glob.glob(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth") if
            #         os.path.isfile(n)]) > 0:
            #     if opt.bgmodel.endswith("plane"):
            #         _, _, _, _, _, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(
            #             train_dataset, visualizer, opt)
            #
            #     resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
            #     if opt.resume_iter == "best":
            #         opt.resume_iter = "latest"
            #     resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
            #     if resume_iter is None:
            #         epoch_count = 1
            #         total_steps = 0
            #         visualizer.print_details("No previous checkpoints, start from scratch!!!!")
            #     else:
            #         opt.resume_iter = resume_iter
            #         states = torch.load(
            #             os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)), map_location=cur_device)
            #         epoch_count = states['epoch_count']
            #         total_steps = states['total_steps']
            #         best_PSNR = states['best_PSNR'] if 'best_PSNR' in states else best_PSNR
            #         best_iter = states['best_iter'] if 'best_iter' in states else best_iter
            #         best_PSNR = best_PSNR.item() if torch.is_tensor(best_PSNR) else best_PSNR
            #         visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            #         visualizer.print_details('Continue training from {} epoch'.format(opt.resume_iter))
            #         visualizer.print_details(f"Iter: {total_steps}")
            #         visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            #         del states
            #     opt.mode = 2
            #     opt.load_points = 1
            #     opt.resume_dir = resume_dir
            #     opt.resume_iter = resume_iter
            #     opt.is_train = True
            #     model = create_model(opt)
            # elif opt.load_points < 1:
            #     points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(
            #         train_dataset, visualizer, opt)
            #     opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            #     opt.is_train = True
            #     opt.mode = 2
            #     model = create_model(opt)
            if opt.load_points == 1:
                load_points = opt.load_points
                opt.is_train = False
                opt.mode = 1
                opt.load_points = 0
                # model = create_model(opt)
                # model.setup(opt)
                # model.eval()
                print('load points:', load_points)
                if load_points in [1, 3]:

                    points_xyz_all = train_dataset.load_init_points()
                    points_xyz_all = points_xyz_all.unsqueeze(0)
                    print('points xyz shape in trainer:', points_xyz_all.shape)
                # if load_points == 2:
                #     points_xyz_all = train_dataset.load_init_depth_points(device="cuda", vox_res=100)
                # if load_points == 3:
                #     depth_xyz_all = train_dataset.load_init_depth_points(device="cuda", vox_res=80)
                #     print("points_xyz_all", points_xyz_all.shape)
                #     print("depth_xyz_all", depth_xyz_all.shape)
                #     filter_res = 100
                #     pc_grid_id, _, pc_space_min, pc_space_max = mvs_utils.construct_vox_points_ind(points_xyz_all,
                #                                                                                    filter_res)
                #     d_grid_id, depth_inds, _, _ = mvs_utils.construct_vox_points_ind(depth_xyz_all, filter_res,
                #                                                                      space_min=pc_space_min,
                #                                                                      space_max=pc_space_max)
                #     all_grid = torch.cat([pc_grid_id, d_grid_id], dim=0)
                #     min_id = torch.min(all_grid, dim=-2)[0]
                #     max_id = torch.max(all_grid, dim=-2)[0] - min_id
                #     max_id_lst = (max_id + 1).cpu().numpy().tolist()
                #     mask = torch.ones(max_id_lst, device=d_grid_id.device)
                #     pc_maskgrid_id = (pc_grid_id - min_id[None, ...]).to(torch.long)
                #     mask[pc_maskgrid_id[..., 0], pc_maskgrid_id[..., 1], pc_maskgrid_id[..., 2]] = 0
                #     depth_maskinds = (d_grid_id[depth_inds, :] - min_id).to(torch.long)
                #     depth_maskinds = mask[depth_maskinds[..., 0], depth_maskinds[..., 1], depth_maskinds[..., 2]]
                #     depth_xyz_all = depth_xyz_all[depth_maskinds > 0]
                #     visualizer.save_neural_points("dep_filtered", depth_xyz_all, None, None, save_ref=False)
                #     print("vis depth; after pc mask depth_xyz_all", depth_xyz_all.shape)
                #     points_xyz_all = [points_xyz_all, depth_xyz_all] if opt.vox_res > 0 else torch.cat(
                #         [points_xyz_all, depth_xyz_all], dim=0)
                #     del depth_xyz_all, depth_maskinds, mask, pc_maskgrid_id, max_id_lst, max_id, min_id, all_grid

                if opt.ranges[0] > -99.0:
                    ranges = torch.as_tensor(opt.ranges, device=points_xyz_all.device, dtype=torch.float32)
                    mask = torch.prod(
                        torch.logical_and(points_xyz_all[..., :3] >= ranges[None, :3],
                                          points_xyz_all[..., :3] <= ranges[None, 3:]),
                        dim=-1) > 0
                    points_xyz_all = points_xyz_all[mask]

                # if opt.vox_res > 0:
                #     points_xyz_all = [points_xyz_all] if not isinstance(points_xyz_all, list) else points_xyz_all
                #     points_xyz_holder = torch.zeros([0, 3], dtype=points_xyz_all[0].dtype, device="cuda")
                #     for i in range(len(points_xyz_all)):
                #         points_xyz = points_xyz_all[i]
                #         vox_res = opt.vox_res // (1.5 ** i)
                #         print("load points_xyz", points_xyz.shape)
                #         _, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(
                #             points_xyz.cuda() if len(points_xyz) < 80000000 else points_xyz[
                #                                                                  ::(len(points_xyz) // 80000000 + 1),
                #                                                                  ...].cuda(), vox_res)
                #         points_xyz = points_xyz[sampled_pnt_idx, :]
                #         print("after voxelize:", points_xyz.shape)
                #         points_xyz_holder = torch.cat([points_xyz_holder, points_xyz], dim=0)
                #     points_xyz_all = points_xyz_holder

                # if opt.resample_pnts > 0:
                #     if opt.resample_pnts == 1:
                #         print("points_xyz_all", points_xyz_all.shape)
                #         inds = torch.min(torch.norm(points_xyz_all, dim=-1, keepdim=True), dim=0)[
                #             1]  # use the point closest to the origin
                #     else:
                #         inds = torch.randperm(len(points_xyz_all))[:opt.resample_pnts, ...]
                #     points_xyz_all = points_xyz_all[inds, ...]

                campos, camdir = train_dataset.get_campos_ray()
                cam_ind = nearest_view(campos, camdir, points_xyz_all, train_dataset.id_list)
                unique_cam_ind = torch.unique(cam_ind)
                print("unique_cam_ind", unique_cam_ind.shape)
                points_xyz_all = [points_xyz_all[cam_ind[:, 0] == unique_cam_ind[i], :] for i in
                                  range(len(unique_cam_ind))]

                featuredim = opt.point_features_dim
                points_embedding_all = torch.zeros([1, 0, featuredim], device=unique_cam_ind.device,
                                                   dtype=torch.float32)
                points_color_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
                points_dir_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
                points_conf_all = torch.zeros([1, 0, 1], device=unique_cam_ind.device, dtype=torch.float32)
                print("extract points embeding & colors", )
                for i in tqdm(range(len(unique_cam_ind))):
                    id = unique_cam_ind[i]
                    batch = train_dataset.get_item(id, full_img=True)
                    HDWD = [train_dataset.height, train_dataset.width]
                    c2w = batch["c2w"][0].cuda()
                    w2c = torch.inverse(c2w)
                    intrinsic = batch["intrinsic"].cuda()
                    # cam_xyz_all 252, 4
                    cam_xyz_all = (torch.cat([points_xyz_all[i], torch.ones_like(points_xyz_all[i][..., -1:])],
                                             dim=-1) @ w2c.transpose(0, 1))[..., :3]
                    embedding, color, dir, conf = self.network.query_embedding(HDWD, cam_xyz_all[None, ...], None,
                                                                        batch['images'].cuda(), c2w[None, None, ...],
                                                                        w2c[None, None, ...], intrinsic[:, None, ...],
                                                                        0, pointdir_w=True)
                    conf = conf * opt.default_conf if opt.default_conf > 0 and opt.default_conf < 1.0 else conf
                    points_embedding_all = torch.cat([points_embedding_all, embedding], dim=1)
                    points_color_all = torch.cat([points_color_all, color], dim=1)
                    points_dir_all = torch.cat([points_dir_all, dir], dim=1)
                    points_conf_all = torch.cat([points_conf_all, conf], dim=1)
                    # visualizer.save_neural_points(id, cam_xyz_all, color, batch, save_ref=True)
                points_xyz_all = torch.cat(points_xyz_all, dim=0)
                # visualizer.save_neural_points("init", points_xyz_all, points_color_all, None, save_ref=load_points == 0)
                # print("vis")
                # visualizer.save_neural_points("cam", campos, None, None, None)
                # print("vis")
                # exit()

                opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
                opt.is_train = True
                opt.mode = 2
                # model = create_model(opt)

            if points_xyz_all is not None:
                if opt.bgmodel.startswith("planepoints"):
                    gen_pnts, gen_embedding, gen_dir, gen_color, gen_conf = train_dataset.get_plane_param_points()
                    # visualizer.save_neural_points("pl", gen_pnts, gen_color, None, save_ref=False)
                    # print("vis pl")
                    points_xyz_all = torch.cat([points_xyz_all, gen_pnts], dim=0)
                    points_embedding_all = torch.cat([points_embedding_all, gen_embedding], dim=1)
                    points_color_all = torch.cat([points_color_all, gen_dir], dim=1)
                    points_dir_all = torch.cat([points_dir_all, gen_color], dim=1)
                    points_conf_all = torch.cat([points_conf_all, gen_conf], dim=1)

                self.network.set_points(points_xyz_all.cuda(), points_embedding_all.cuda(),
                                 points_color=points_color_all.cuda(),
                                 points_dir=points_dir_all.cuda(), points_conf=points_conf_all.cuda(),
                                 Rw2c=normRw2c.cuda() if opt.load_points < 1 and opt.normview != 3 else None)
                epoch_count = 1
                total_steps = 0
                print(points_embedding_all)

                print(points_embedding_all.shape)
                del points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all


    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()


        best_para, start_step = self._load_model()
        train_iter = iter(self.train_set)

        pbar = tqdm(total=self.cfg['total_step'], bar_format='{r_bar}')
        pbar.update(start_step)

        # Here we load points embeddings
        self.load_init_points()

        for step in range(start_step, self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step'] = step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            # if (step + 1) % self.cfg['novel_view_interval'] == 0:
            #     render_data = train_data.copy()
            #     render_data["render"] = True

            #     self.train_network(render_data)

            log_info = {}
            outputs = self.train_network(train_data)
            for loss in self.train_losses:
                loss_results = loss(outputs, train_data, step)
                for k, v in loss_results.items():
                    log_info[k] = v

            loss = 0
            for k, v in log_info.items():
                if k.startswith('loss'):
                    loss = loss + torch.mean(v)

            loss.backward()
            self.optimizer.step()
            if ((step + 1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, step + 1, 'train')

            if (step + 1) % self.cfg['val_interval'] == 0 or (step + 1) == self.cfg['total_step']:
                torch.cuda.empty_cache()
                val_results = {}
                val_para = 0
                for vi, val_set in enumerate(self.val_set_list):
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        val_results[f'{self.val_set_names[vi]}-{k}'] = v
                    # always use the final val set to select model!
                    val_para = val_para_cur

                if val_para > best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para = val_para
                    self._save_model(step + 1, best_para, self.best_pth_fn)
                self._log_data(val_results, step + 1, 'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step + 1) % self.cfg['save_interval'] == 0:
                save_fn = None
                self._save_model(step + 1, best_para, save_fn=save_fn)

            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()), lr=lr)
            pbar.update(1)
            del loss, log_info

        pbar.close()

    def _load_model(self):
        best_para, start_step = 0, 0
        if os.path.exists(self.pth_fn):
            checkpoint = torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step': step,
            'best_para': best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir)

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results = {}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = v
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)
