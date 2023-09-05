dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': '/datasets/ffhq512_eg3d/train',
	'celeba_test': '/datasets/ffhq512_eg3d/val',
	'ffhq_camera': '/datasets/ffhq512_eg3d/dataset.json',
	'ffhq_pose': '/datasets/ffhq512_eg3d/cameras.json',
	'celeba_test_camera': '/datasets/ffhq512_eg3d/dataset.json',
	'celeba_test_pose': '/datasets/ffhq512_eg3d/cameras.json',

	#  Cats Dataset (In the paper: AFHQ Cat)
	'cats_train': '/datasets/afhq_v2/train/mirrored_cat',
	'cats_test': '/datasets/afhq_v2/val/cat',
	'cats_camera': '/datasets/afhq_v2/dataset_cats.json',
	'cats_test_camera': '/datasets/afhq_v2/dataset_cats.json',
}

model_paths = {
	'eg3d_ffhq': './pretrained/eg3d_G_ema.pkl',
	'ir_se50': './pretrained/model_ir_se50.pth',
	'shape_predictor': './pretrained/shape_predictor_68_face_landmarks.dat',
	'moco': './pretrained/moco_v2_800ep_pretrain.pth'
}
