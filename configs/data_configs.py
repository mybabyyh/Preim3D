from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
		'train_camera_param_data': dataset_paths['ffhq_camera'],
		'train_pose_param_data': dataset_paths['ffhq_pose'],
		'test_camera_param_data': dataset_paths['celeba_test_camera'],
		'test_pose_param_data': dataset_paths['celeba_test_pose']
	},
	'cats_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['cats_train'],
		'train_target_root': dataset_paths['cats_train'],
		'test_source_root': dataset_paths['cats_test'],
		'test_target_root': dataset_paths['cats_test'],
		'train_camera_param_data': dataset_paths['cats_camera'],
		'test_camera_param_data': dataset_paths['cats_test_camera']
	}
}
