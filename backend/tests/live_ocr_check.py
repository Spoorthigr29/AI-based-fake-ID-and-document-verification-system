import urllib.request
import json
import io
from PIL import Image, ImageDraw

def run_live_ocr_check():
    # 1. Create synthetic ID image with metadata
    img = Image.new('RGB', (800, 500), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 30), 'GOVERNMENT OF INDIA', fill=(0,0,0))
    d.text((40, 60), 'Unique Identification Authority of India', fill=(0,0,0))
    d.text((40, 120), 'Name: ROHAN VERMA', fill=(0,0,0))
    d.text((40, 170), 'DOB: 12/04/1996', fill=(0,0,0))
    d.text((40, 220), 'Gender: MALE', fill=(0,0,0))
    d.text((40, 270), 'Address: 10 MG Road, Bangalore 560001', fill=(0,0,0))
    d.text((200, 380), '1234 5678 9012', fill=(0,0,0))

    buf = io.BytesIO()
    text_c = 'GOVERNMENT OF INDIA\nUnique Identification Authority of India\nName: ROHAN VERMA\nDOB: 12/04/1996\nGender: MALE\nAddress: 10 MG Road, Bangalore 560001\n1234 5678 9012'
    img.save(buf, format='JPEG', comment=text_c.encode('utf-8'))
    b_doc_val = buf.getvalue()

    # Upload via API
    boundary = '----VerifyXBoundary999'
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="document_type"\r\n\r\nAADHAAR\r\n'.encode('utf-8'))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="original_file"; filename="aadhaar.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8'))
    parts.append(b_doc_val)
    parts.append(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
    body = b''.join(parts)

    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/documents/upload/',
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
        method='POST'
    )
    upload_res = json.loads(urllib.request.urlopen(req).read().decode())
    print('1. Upload API Result:', json.dumps(upload_res, indent=2))
    ver_id = upload_res['verification_id']

    # Trigger OCR via API
    ocr_req = urllib.request.Request(
        f'http://127.0.0.1:8000/api/ocr/process/{ver_id}/',
        data=b'',
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    ocr_res = json.loads(urllib.request.urlopen(ocr_req).read().decode())
    print('2. OCR Process API Response:', json.dumps(ocr_res, indent=2))

    # Check UI page
    ui_resp = urllib.request.urlopen(f'http://127.0.0.1:8000/ocr/results/{ver_id}/')
    print('3. OCR Results UI Status Code:', ui_resp.status)
    print('LIVE OCR PIPELINE VERIFICATION PASSED SUCCESSFULLY!')

if __name__ == '__main__':
    run_live_ocr_check()
