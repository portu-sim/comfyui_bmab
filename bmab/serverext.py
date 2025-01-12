from server import PromptServer
from aiohttp import web

from PIL import Image
from bmab.nodes import fill, upscaler
import base64
from io import BytesIO
from bmab import utils


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
	j = await request.json()
	b64img = j.get('image')
	model = j.get('model')
	scale = j.get('scale', '2')
	width = j.get('width', 0)
	height = j.get('height', 0)

	up = upscaler.BMABUpscaleWithModel()
	img = b64_decoding(b64img)
	if scale != 0:
		width, height = int(img.width * scale), int(img.height * scale)
	pixels = utils.get_pixels_from_pils([img])
	s = up.upscale_with_model(model, pixels, progress=False)
	pil_images = utils.get_pils_from_pixels(s)
	result = pil_images[0].resize((width, height), Image.Resampling.LANCZOS)
	data = {'image': b64_encoding(result)}
	return web.json_response(data)


@PromptServer.instance.routes.get("/bmab/images")
async def bmab_upscale(request):
	remote_client_id = request.rel_url.query.get("clientId")
	imgs = memory_image_storage.get(remote_client_id, [])
	results = [b64_encoding(i) for i in imgs]
	data = {'images': results}
	return web.json_response(data)


