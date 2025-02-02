from server import PromptServer
from aiohttp import web

from PIL import Image
from bmab.nodes import fill, upscaler
import base64
from io import BytesIO
from bmab import utils
from comfy import utils as cutils
from bmab.utils import yolo, sam

memory_image_storage = {}


def b64_encoding(image):
	buffered = BytesIO()
	image.save(buffered, format="PNG")
	return base64.b64encode(buffered.getvalue()).decode("utf-8")


def b64_decoding(b64):
	return Image.open(BytesIO(base64.b64decode(b64)))


client_ids = {}


@PromptServer.instance.routes.get("/bmab")
async def bmab_register_client(request):
	remote_client_id = request.rel_url.query.get("remote_client_id")
	remote_name = request.rel_url.query.get("remote_name")

	print(remote_client_id, remote_name)
	client_ids[remote_client_id] = {'name': remote_name}
	data = {'name': remote_name, 'client_id': remote_client_id}
	return web.json_response(data)


@PromptServer.instance.routes.get("/bmab/remote")
async def bmab_remote(request):
	command = request.rel_url.query.get("command")
	name = request.rel_url.query.get("name")
	data = {'command': command, 'name': name}

	if command == 'queue':
		for sid, v in client_ids.items():
			if v.get('name') == name:
				await PromptServer.instance.send("bmab_queue", {"status": '', 'sid': sid}, sid)
				data['client_id'] = sid

	return web.json_response(data)


@PromptServer.instance.routes.post("/bmab/outpaintbyratio")
async def bmab_outpaintbyratio(request):
	j = await request.json()
	b64img = j.get('image')
	if b64img is not None:
		prompt = j.get('prompt')
		align = j.get('align', 'bottom')
		ratio = j.get('ratio', 0.85)
		dilation = j.get('dilation', 16)
		steps = j.get('steps', 8)

		filler = fill.BMABOutpaintByRatio()
		img = filler.infer(b64_decoding(b64img), align, ratio, dilation, steps, prompt)
		data = {'image': b64_encoding(img)}
		return web.json_response(data)
	else:
		print('release')
		fill.unload()
		return web.json_response({})


@PromptServer.instance.routes.post("/bmab/reframe")
async def bmab_reframe(request):
	j = await request.json()
	b64img = j.get('image')
	if b64img is not None:
		prompt = j.get('prompt')
		ratio = j.get('ratio', '1:1')
		dilation = j.get('dilation', 16)
		steps = j.get('steps', 8)
		r = fill.BMABReframe.ratio_sel.get(ratio, (1024, 1024))

		filler = fill.BMABReframe()
		img = filler.infer(b64_decoding(b64img), r[0], r[1], dilation, steps, prompt)
		data = {'image': b64_encoding(img)}
		return web.json_response(data)
	else:
		print('release')
		fill.unload()
		return web.json_response({})


@PromptServer.instance.routes.post("/bmab/upscale")
async def bmab_upscale(request):
	from comfy import utils as cutils

	j = await request.json()
	b64img = j.get('image')
	model = j.get('model')
	scale = j.get('scale', '2')
	width = j.get('width', 0)
	height = j.get('height', 0)

	hook = cutils.PROGRESS_BAR_HOOK
	cutils.PROGRESS_BAR_HOOK = None
	try:
		up = upscaler.BMABUpscaleWithModel()
		img = b64_decoding(b64img)
		if scale != 0:
			width, height = int(img.width * scale), int(img.height * scale)
		pixels = utils.get_pixels_from_pils([img])
		s = up.upscale_with_model(model, pixels, progress=False)
		utils.torch_gc()
	finally:
		cutils.PROGRESS_BAR_HOOK = hook

	pil_images = utils.get_pils_from_pixels(s)
	result = pil_images[0].resize((width, height), Image.Resampling.LANCZOS)
	data = {'image': b64_encoding(result)}
	return web.json_response(data)


@PromptServer.instance.routes.post("/bmab/inpaint")
async def bmab_inpaint(request):
	j = await request.json()
	b64img = j.get('image')
	if b64img is not None:
		b64mask = j.get('mask')
		prompt = j.get('prompt')
		steps = j.get('steps')

		inpaint = fill.BMABInpaint()
		img = b64_decoding(b64img)
		msk = b64_decoding(b64mask).convert('L')

		result = inpaint.infer(img, msk, steps, prompt_input=prompt)
		data = {'image': b64_encoding(result)}
		return web.json_response(data)
	else:
		print('release')
		fill.unload()
		return web.json_response({})


@PromptServer.instance.routes.post("/bmab/depth")
async def bmab_inpaint(request):
	from custom_nodes.comfyui_controlnet_aux.node_wrappers.depth_anything import Depth_Anything_Preprocessor

	j = await request.json()
	b64img = j.get('image')
	resolution = j.get('resolution')
	images = utils.get_pixels_from_pils([b64_decoding(b64img)])
	hook = cutils.PROGRESS_BAR_HOOK
	cutils.PROGRESS_BAR_HOOK = None
	try:
		node = Depth_Anything_Preprocessor()
		out = node.execute(images, 'depth_anything_vitl14.pth', resolution)
	finally:
		cutils.PROGRESS_BAR_HOOK = hook
	pil_images = utils.get_pils_from_pixels(out[0])
	data = {'image': b64_encoding(pil_images[0])}
	return web.json_response(data)


@PromptServer.instance.routes.post("/bmab/sam")
async def bmab_sam(request):
	j = await request.json()
	b64img = j.get('image')
	model = j.get('model')
	confidence = j.get('confidence')

	image = b64_decoding(b64img)
	boxes, conf = yolo.predict(image, model, confidence)
	for box in boxes:
		mask = sam.sam_predict_box(image, box)
		data = {'image': b64_encoding(mask.convert('RGB'))}
		sam.release()
		return web.json_response(data)
	return web.json_response({})


@PromptServer.instance.routes.post("/bmab/detect")
async def bmab_detect(request):
	j = await request.json()
	b64img = j.get('image')
	model = j.get('model')
	confidence = j.get('confidence')

	image = b64_decoding(b64img)
	boxes, conf = yolo.predict(image, model, confidence)
	data = {'boxes': boxes}
	return web.json_response(data)


@PromptServer.instance.routes.post("/bmab/free")
async def bmab_free(request):
	fill.unload()
	return web.json_response({})


@PromptServer.instance.routes.get("/bmab/images")
async def bmab_upscale(request):
	remote_client_id = request.rel_url.query.get("clientId")
	imgs = memory_image_storage.get(remote_client_id, [])
	results = [b64_encoding(i) for i in imgs]
	data = {'images': results}
	return web.json_response(data)


from transformers import AutoProcessor, AutoModelForCausalLM
import torch


@PromptServer.instance.routes.post("/bmab/caption")
async def bmab_upscale(request):
	j = await request.json()
	b64img = j.get('image')

	image = b64_decoding(b64img)
	caption = run_captioning(image)
	ret = {
		'caption': caption
	}
	return web.json_response(ret)


def run_captioning(image):
	print(f"run_captioning")
	# Load internally to not consume resources for training
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"device={device}")
	torch_dtype = torch.float16
	model = AutoModelForCausalLM.from_pretrained(
		"multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
	).to(device)
	processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

	prompt = "<DETAILED_CAPTION>"
	inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
	print(f"inputs {inputs}")

	generated_ids = model.generate(
		input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
	)
	print(f"generated_ids {generated_ids}")

	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
	print(f"generated_text: {generated_text}")
	parsed_answer = processor.post_process_generation(
		generated_text, task=prompt, image_size=(image.width, image.height)
	)
	print(f"parsed_answer = {parsed_answer}")
	caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")


	model.to("cpu")
	del model
	del processor
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return caption_text