from server import PromptServer
from aiohttp import web

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
