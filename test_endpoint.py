import urllib.request
import json
data = json.dumps({"wing_type":"ga","span":11,"ar":7.2,"taper":0.45,"sweep_deg":3,"altitude":2000,"velocity":55,"thickness":0.12,"camber":0.02}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/analyze', data=data, headers={'Content-Type': 'application/json'})
try:
    print(urllib.request.urlopen(req).read().decode())
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code, e.read().decode())
