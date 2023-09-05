from PIL import Image
import matplotlib.pyplot as plt


# Log images
def log_input_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')


# inference result visual
def inference_vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(14, 4 * display_count))
	gs = fig.add_gridspec(display_count, 5)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		inference_vis_faces_with_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def inference_vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\n{}={:.3f}'.format(hooks_dict['att_name'], float(hooks_dict['att_score_input'])))

	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['proj_face'])
	plt.title('Projection\nID={:.3f}, {}={:.3f}'.format(float(hooks_dict['arc_sim_proj']), hooks_dict['att_name'],
														float(hooks_dict['att_score_proj'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['left_face'])
	plt.title('Left\nID={:.3f}, {}={:.3f}'.format(float(hooks_dict['arc_sim_left']), hooks_dict['att_name'],
														float(hooks_dict['att_score_left'])))
	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['center_face'])
	plt.title('Center\nID={:.3f}, {}={:.3f}'.format(float(hooks_dict['arc_sim_center']), hooks_dict['att_name'],
														float(hooks_dict['att_score_center'])))
	fig.add_subplot(gs[i, 4])
	plt.imshow(hooks_dict['right_face'])
	plt.title('Right\nID={:.3f}, {}={:.3f}'.format(float(hooks_dict['arc_sim_right']), hooks_dict['att_name'],
														float(hooks_dict['att_score_right'])))