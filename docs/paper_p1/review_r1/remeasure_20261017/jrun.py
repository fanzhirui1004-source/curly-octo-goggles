"""Run code on the AutoDL Jupyter server through the kernel API.

Usage:  jrun.py sh  '<shell command>' [timeout_s]     run a shell command on the server, print stdout/stderr
        jrun.py py  '<python code>'    [timeout_s]     run python code in the kernel
        jrun.py stop                                   shut the kernel down
Reads JH (host) and JT (token) from the environment; the kernel id is kept in jkernel.txt next to this file.
"""
import json, os, sys, time, uuid
import requests, websocket

HERE = os.path.dirname(os.path.abspath(__file__))
KFILE = os.path.join(HERE, 'jkernel.txt')
JH, JT = os.environ['JH'], os.environ['JT']
BASE = JH + '/jupyter'
HDR = {'Authorization': 'token ' + JT}
CA = os.environ.get('SSL_CERT_FILE', '/root/.ccr/ca-bundle.crt')


def kernel_id():
    if os.path.exists(KFILE):
        kid = open(KFILE).read().strip()
        r = requests.get(f'{BASE}/api/kernels/{kid}', headers=HDR, timeout=30)
        if r.status_code == 200:
            return kid
    r = requests.post(f'{BASE}/api/kernels', headers=HDR, json={'name': 'python3'}, timeout=60)
    r.raise_for_status()
    kid = r.json()['id']
    with open(KFILE, 'w') as f:
        f.write(kid)
    time.sleep(2)
    return kid


def connect(kid):
    url = BASE.replace('https://', 'wss://') + f'/api/kernels/{kid}/channels'
    px = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    kw = {}
    if px:
        hp = px.split('://', 1)[-1].rstrip('/')
        host, port = hp.rsplit(':', 1)
        kw = dict(http_proxy_host=host, http_proxy_port=int(port), proxy_type='http')
    return websocket.create_connection(url, header=[f'Authorization: token {JT}'], sslopt={'ca_certs': CA},
                                       timeout=30, **kw)


def execute(code, timeout):
    kid = kernel_id()
    ws = connect(kid)
    msg_id = uuid.uuid4().hex
    msg = {'header': {'msg_id': msg_id, 'username': 'claude', 'session': uuid.uuid4().hex, 'msg_type': 'execute_request',
                      'version': '5.3', 'date': ''},
           'parent_header': {}, 'metadata': {}, 'channel': 'shell',
           'content': {'code': code, 'silent': False, 'store_history': False, 'user_expressions': {}, 'allow_stdin': False,
                       'stop_on_error': True}}
    ws.send(json.dumps(msg))
    t0 = time.time()
    out = []
    while True:
        if time.time() - t0 > timeout:
            out.append(f'\n[jrun: client timeout after {timeout}s; the code may still be running on the server]\n')
            break
        try:
            m = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            continue
        if m.get('parent_header', {}).get('msg_id') != msg_id:
            continue
        t, c = m.get('msg_type'), m.get('content', {})
        if t == 'stream':
            out.append(c.get('text', ''))
        elif t in ('execute_result', 'display_data'):
            out.append(c.get('data', {}).get('text/plain', '') + '\n')
        elif t == 'error':
            out.append('\n'.join(c.get('traceback', [])) + '\n')
        elif t == 'status' and c.get('execution_state') == 'idle':
            break
    ws.close()
    sys.stdout.write(''.join(out))


def main():
    mode = sys.argv[1]
    if mode == 'stop':
        if os.path.exists(KFILE):
            kid = open(KFILE).read().strip()
            r = requests.delete(f'{BASE}/api/kernels/{kid}', headers=HDR, timeout=30)
            print('kernel stopped', r.status_code)
            os.remove(KFILE)
        return
    arg = sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 120
    if mode == 'sh':
        code = ('import subprocess\n'
                f'r = subprocess.run({arg!r}, shell=True, capture_output=True, text=True, timeout={timeout - 5})\n'
                'print(r.stdout, end="")\n'
                'print(r.stderr, end="")\n')
    else:
        code = arg
    execute(code, timeout)


if __name__ == '__main__':
    main()
