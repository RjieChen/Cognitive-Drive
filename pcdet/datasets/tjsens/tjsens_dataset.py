import copy
import pickle
from pathlib import Path 

import numpy as np
import cv2
import torch

from skimage import io
from . import kitti_utils
from ...ops.iou3d_nms import iou3d_nms_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_tjsens
from ..dataset import DatasetTemplate

class TjsensDataset(DatasetTemplate): 
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')

        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.tjsens_infos = [] 
        self.include_tjsens_data(self.mode)

    def include_tjsens_data(self, mode): 
        if self.logger is not None:
            self.logger.info('Loading TJSens dataset')
        tjsens_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                tjsens_infos.extend(infos)

        self.tjsens_infos.extend(tjsens_infos)

        if self.logger is not None:
            self.logger.info('Total samples for TJSens dataset: %d' % (len(tjsens_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None


    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.jpeg' % idx) 
        if not img_file.exists(): 
            img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists(), f"Image file not found for index {idx}: tried .jpeg and .jpg"
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.jpeg' % idx)
        if not img_file.exists():
            img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_tjsens.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)


    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.array([obj.box2d for obj in obj_list], dtype=np.float32)
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
                
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                    

                clean_gt_boxes_lidar_list = []

                for obj in obj_list:
                    loc_camera_homogeneous = np.ones(4)
                    loc_camera_homogeneous[:3] = obj.loc  

                    tr_v2c_4x4 = np.eye(4)
                    tr_v2c_4x4[:3, :] = calib.V2C
                    tr_c2v_4x4 = np.linalg.inv(tr_v2c_4x4)

                    loc_lidar_homogeneous = tr_c2v_4x4 @ loc_camera_homogeneous
                    cx_lidar, cy_lidar, cz_lidar = loc_lidar_homogeneous[:3]

                    from scipy.spatial.transform import Rotation as R
                    rot = R.from_euler('XYZ', [obj.rx, obj.ry, obj.rz], degrees=False)
                    R_obj_in_cam = rot.as_matrix()

                    R_cam_from_lidar = calib.V2C[:3, :3]

                    R_lidar_from_cam = R_cam_from_lidar.T
                    R_obj_in_lidar = R_lidar_from_cam @ R_obj_in_cam

                    yaw_lidar_raw = np.arctan2(R_obj_in_lidar[1, 0], R_obj_in_lidar[0, 0])

                    if yaw_lidar_raw > np.pi / 2:
                        yaw_lidar_raw -= 2 * np.pi

                    l, w, h = obj.l, obj.w, obj.h

                    gt_box_lidar = np.array([cx_lidar, cy_lidar, cz_lidar, l, w, h, yaw_lidar_raw])
                    clean_gt_boxes_lidar_list.append(gt_box_lidar)

                gt_boxes_lidar = np.array(clean_gt_boxes_lidar_list, dtype=np.float32)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                

                gt_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes_lidar, calib)

                annotations['location'] = gt_boxes_camera[:, 0:3] 
                annotations['dimensions'] = gt_boxes_camera[:, 3:6][:, [1, 2, 0]] # lhw -> hwl
                annotations['rotation_y'] = gt_boxes_camera[:, 6]

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar_gt = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_objects, dtype=np.int32)
                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar_gt[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('tjsens_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # 1. 从PyTorch Tensor中提取核心预测数据，并转为NumPy数组
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes_lidar = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            if pred_labels.shape[0] == 0:
                single_pred_dict = {
                    'frame_id': batch_dict['frame_id'][index],
                    'name': np.array([]),
                    'score': np.array([]),
                    'boxes_lidar': np.zeros((0, 7))
                }
                annos.append(single_pred_dict)
                continue

            pred_names = np.array(class_names)[pred_labels - 1]
            single_pred_dict = {
                'frame_id': batch_dict['frame_id'][index],
                'name': pred_names,
                'score': pred_scores,
                'boxes_lidar': pred_boxes_lidar  
            }
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.tjsens_infos[0].keys():
            return 'No gt annotations in info file, skip evaluation', {}
        
        logger = kwargs.get('logger', None)

        pred_dicts = det_annos
        gt_dicts = [info['annos'] for info in self.tjsens_infos]

        if logger:
            logger.info("--- Starting Pure LiDAR Coordinate Evaluation (Overall AP) ---")
        else:
            print("--- Starting Pure LiDAR Coordinate Evaluation (Overall AP) ---")
        
        iou_thresholds_dict = {
            'Car': [0.7, 0.5], 'Pedestrian': [0.5, 0.3], 'Cyclist': [0.5, 0.3], 'Truck': [0.7, 0.5]
        }
        
        final_ap_dict = {}
        ap_result_str = ''

        for class_name in class_names:
            if class_name not in iou_thresholds_dict:
                continue
            
            class_preds_list = []
            class_gts_list = []
            for i in range(len(pred_dicts)):
                pred_mask = pred_dicts[i]['name'] == class_name
                class_preds_list.append({
                    'pred_boxes': pred_dicts[i]['boxes_lidar'][pred_mask],
                    'pred_scores': pred_dicts[i]['score'][pred_mask]
                })
                
                gt_mask = gt_dicts[i]['name'] == class_name
                class_gts_list.append(gt_dicts[i]['gt_boxes_lidar'][gt_mask])

            for iou_thresh in iou_thresholds_dict[class_name]:
                # BEV AP
                ap_bev_11, ap_bev_40 = self.calculate_map(
                    class_preds_list, class_gts_list, iou_thresh, metric='bev'
                )
                # 3D AP
                ap_3d_11, ap_3d_40 = self.calculate_map(
                    class_preds_list, class_gts_list, iou_thresh, metric='3d'
                )

                ap_result_str += f'\n{class_name} AP @ IoU={iou_thresh:.2f}:\n'
                ap_result_str += 'bev  AP (11-point): {:.2f}\n'.format(ap_bev_11)
                ap_result_str += '3d   AP (11-point): {:.2f}\n'.format(ap_3d_11)
                ap_result_str += 'bev  AP_R40:        {:.2f}\n'.format(ap_bev_40)
                ap_result_str += '3d   AP_R40:        {:.2f}\n'.format(ap_3d_40)

                final_ap_dict[f'{class_name}_BEV_R40_@{iou_thresh}'] = ap_bev_40
                final_ap_dict[f'{class_name}_3D_R40_@{iou_thresh}'] = ap_3d_40

        return ap_result_str, final_ap_dict


    @staticmethod
    def calculate_map(pred_dicts, gt_boxes_list, iou_thresh, metric='3d'):

        num_samples = len(pred_dicts)
        
        preds_info = []
        for i in range(num_samples):
            for j in range(len(pred_dicts[i]['pred_scores'])):
                preds_info.append({
                    "frame_idx": i, "score": pred_dicts[i]['pred_scores'][j], "box": pred_dicts[i]['pred_boxes'][j]
                })
        preds_info.sort(key=lambda x: x['score'], reverse=True)
        num_preds = len(preds_info)

        gt_matched_flags = [np.zeros(len(gts), dtype=bool) for gts in gt_boxes_list]
        total_num_gts = sum(len(gts) for gts in gt_boxes_list)
        if total_num_gts == 0:
            return 0.0, 0.0

        tp_flags = np.zeros(num_preds, dtype=bool)
        
        for i in range(num_preds):
            pred_info = preds_info[i]
            frame_idx = pred_info['frame_idx']
            pred_box = pred_info['box'].reshape(1, -1)
            
            gt_boxes_in_frame = gt_boxes_list[frame_idx]
            
            if len(gt_boxes_in_frame) == 0:
                continue

            pred_box_torch = torch.from_numpy(pred_box).float().cuda()
            gt_boxes_torch = torch.from_numpy(gt_boxes_in_frame).float().cuda()
            
            if metric == 'bev':
                iou_vector = iou3d_nms_utils.boxes_iou_bev(pred_box_torch, gt_boxes_torch).cpu().numpy()[0]
            else: # 3d
                iou_vector = iou3d_nms_utils.boxes_iou3d_gpu(pred_box_torch, gt_boxes_torch).cpu().numpy()[0] # <--- 修复了这里的拼写错误
                
            best_gt_idx = np.argmax(iou_vector)
            
            max_iou = float(iou_vector[best_gt_idx])
            is_matched = bool(gt_matched_flags[frame_idx][best_gt_idx])

            if max_iou >= iou_thresh and not is_matched:
                tp_flags[i] = True
                gt_matched_flags[frame_idx][best_gt_idx] = True

        tp_cumsum = np.cumsum(tp_flags)
        fp_cumsum = np.cumsum(~tp_flags)
        
        recalls = tp_cumsum / total_num_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # AP_11
        ap_11 = 0.0
        for t in np.arange(0, 1.1, 0.1):
            p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0
            ap_11 += p / 11.0
        ap_11 *= 100

        # AP_40
        ap_40 = 0.0
        recall_levels = np.linspace(0, 1.0, num=41, endpoint=True)[1:]
        for t in recall_levels:
            p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0
            ap_40 += p / 40.0
        ap_40 *= 100

        return ap_11, ap_40


    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.tjsens_infos) * self.total_epochs
        return len(self.tjsens_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.tjsens_infos)
        info = copy.deepcopy(self.tjsens_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_boxes_lidar = annos['gt_boxes_lidar']

            # loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            # gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            # gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]

            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        input_dict['calib'] = calib
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict
        

def create_tjsens_infos(dataset_cfg, class_names, data_path, save_path, workers=4): # <--- 3. 函数名修改
    dataset = TjsensDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('tjsens_infos_%s.pkl' % train_split)
    val_filename = save_path / ('tjsens_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'tjsens_infos_trainval.pkl'
    test_filename = save_path / 'tjsens_infos_test.pkl'

    print('---------------Start to generate data infos for TJSens---------------')

    dataset.set_split(train_split)
    tjsens_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(tjsens_infos_train, f)
    print('TJSens info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    tjsens_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(tjsens_infos_val, f)
    print('TJSens info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(tjsens_infos_train + tjsens_infos_val, f)
    print('TJSens info trainval file is saved to %s' % trainval_filename)
    
    dataset.set_split('test')
    tjsens_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(tjsens_infos_test, f)
    print('Tjsens info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split) # create_groundtruth_database 内部也需要修改 'kitti' -> 'tjsens'

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_tjsens_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        
        DATA_PATH = ROOT_DIR / 'data' / 'tjsens' 
        CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Truck'] 
        
        create_tjsens_infos( 
            dataset_cfg=dataset_cfg,
            class_names=CLASS_NAMES,
            data_path=DATA_PATH,
            save_path=DATA_PATH
        )
