#!/usr/bin/env python3
"""Run a shell command on the AutoDL box through the Jupyter kernel websocket."""
import os, sys, json, uuid, ssl, urllib.request, websocket
HOST = os.environ["JHOST"]
TOK = os.environ["JTOK"]
BASE = f"https://{HOST}/jupyter"
HDR = {"Authorization": f"token {TOK}"}
PROXY = os.environ["HTTPS_PROXY"].replace("http://", "").split(":")
CA = "/root/.ccr/ca-bundle.crt"

def api(path, method="GET", body=None):
    req = urllib.request.Request(BASE + path, method=method, headers=HDR | ({"Content-Type": "application/json"} if body else {}),
                                 data=json.dumps(body).encode() if body else None)
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context(cafile=CA)) as r:
        return json.loads(r.read() or b"null")

def get_kernel():
    if os.environ.get("JRUN_NEW_KERNEL"): return api("/api/kernels", "POST", {"name": "python3"})["id"]
    ks = api("/api/kernels")
    py = [k for k in ks if k.get("name", "").startswith("python")]
    for k in py:                                                   # an idle kernel first: a busy one queues the command
        if k.get("execution_state") == "idle": return k["id"]
    if py: return py[0]["id"]
    return api("/api/kernels", "POST", {"name": "python3"})["id"]

def run(cmd, timeout=3600):
    kid = get_kernel()
    ws = websocket.create_connection(f"wss://{HOST}/jupyter/api/kernels/{kid}/channels", header=[f"Authorization: token {TOK}"],
                                     http_proxy_host=PROXY[0], http_proxy_port=int(PROXY[1]), proxy_type="http",
                                     sslopt={"ca_certs": CA}, timeout=timeout)
    code = "import subprocess,sys\nr=subprocess.run(%r,shell=True,capture_output=True)\nsys.stdout.write(r.stdout.decode('utf-8','replace'));sys.stderr.write(r.stderr.decode('utf-8','replace'));print('__RC__',r.returncode)" % cmd
    mid = uuid.uuid4().hex
    ws.send(json.dumps({"header": {"msg_id": mid, "username": "u", "session": uuid.uuid4().hex, "msg_type": "execute_request", "version": "5.3"},
                        "parent_header": {}, "metadata": {}, "channel": "shell",
                        "content": {"code": code, "silent": False, "store_history": False, "allow_stdin": False, "stop_on_error": True}}))
    out, err, rc = [], [], None
    while True:
        m = json.loads(ws.recv())
        if m.get("parent_header", {}).get("msg_id") != mid: continue
        t = m["msg_type"]
        if t == "stream":
            (out if m["content"]["name"] == "stdout" else err).append(m["content"]["text"])
        elif t == "error":
            err.append("\n".join(m["content"]["traceback"]))
        elif t == "execute_reply":
            break
    ws.close()
    o = "".join(out)
    if "__RC__" in o:
        o, tail = o.rsplit("__RC__", 1); rc = int(tail.strip() or 1)
    return rc if rc is not None else 1, o, "".join(err)

if __name__ == "__main__":
    rc, o, e = run(" ".join(sys.argv[1:]) or "hostname")
    sys.stdout.write(o); sys.stderr.write(e); sys.exit(rc)
