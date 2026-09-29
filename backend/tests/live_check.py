import urllib.request
import json
import io
from PIL import Image

def run_live_check():
    # 1. Test GET routes
    for path in ['/verification/', '/verification/upload/']:
        resp = urllib.request.urlopen(f'http://127.0.0.1:8000{path}')
        print(f'GET {path}: {resp.status}')

    # 2. Test API Upload via Multipart
    img = Image.new('RGB', (120, 120), color='teal')
    b_doc = io.BytesIO()
    img.save(b_doc, format='JPEG')
    b_doc_val = b_doc.getvalue()

    selfie = Image.new('RGB', (100, 100), color='salmon')
    b_selfie = io.BytesIO()
    selfie.save(b_selfie, format='PNG')
    b_selfie_val = b_selfie.getvalue()

    boundary = '----VerifyXBoundary123456'
    
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="document_type"\r\n\r\nPASSPORT\r\n'.encode('utf-8'))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="original_file"; filename="passport.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8'))
    parts.append(b_doc_val)
    parts.append(f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="selfie_file"; filename="selfie.png"\r\nContent-Type: image/png\r\n\r\n'.encode('utf-8'))
    parts.append(b_selfie_val)
    parts.append(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
    
    body = b''.join(parts)

    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/documents/upload/',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )

    res = urllib.request.urlopen(req)
    res_data = json.loads(res.read().decode())
    print('POST /api/documents/upload/ status:', res.status)
    print('Response Payload:', json.dumps(res_data, indent=2))

    # 3. Test Detail Page
    ver_id = res_data['verification_id']
    detail_resp = urllib.request.urlopen(f'http://127.0.0.1:8000/verification/{ver_id}/')
    print(f'GET /verification/{ver_id}/ status: {detail_resp.status}')
    print("ALL LIVE CHECKS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    run_live_check()
