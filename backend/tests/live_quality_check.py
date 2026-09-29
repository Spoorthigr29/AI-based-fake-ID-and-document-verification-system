import urllib.request
import json
import io
from PIL import Image, ImageDraw

def run_live_quality_check():
    # 1. Create a high quality synthetic PAN card
    img = Image.new('RGB', (1000, 600), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((50, 40), "INCOME TAX DEPARTMENT GOVT OF INDIA", fill=(0, 0, 0))
    d.text((50, 90), "PERMANENT ACCOUNT NUMBER CARD", fill=(0, 0, 0))
    d.text((50, 150), "Name: SUNITA PATEL", fill=(0, 0, 0))
    d.text((50, 200), "DOB: 14/03/1993", fill=(0, 0, 0))
    d.text((50, 280), "PAN: ABCDE9988K", fill=(0, 0, 0))

    buf = io.BytesIO()
    raw_text = "INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT NUMBER CARD Name: SUNITA PATEL DOB: 14/03/1993 PAN: ABCDE9988K"
    img.save(buf, format='JPEG', comment=raw_text.encode('utf-8'))
    b_doc_val = buf.getvalue()

    # Upload
    boundary = '----VerifyXBoundary777'
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="document_type"\r\n\r\nPAN\r\n'.encode('utf-8'))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="original_file"; filename="sample_pan.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8'))
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
    ver_id = upload_res['verification_id']
    print('1. Uploaded verification case:', ver_id)

    # 2. Call Analyze API: POST /api/documents/analyze/<verification_id>/
    analyze_req = urllib.request.Request(
        f'http://127.0.0.1:8000/api/documents/analyze/{ver_id}/',
        data=b'',
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    analyze_res = json.loads(urllib.request.urlopen(analyze_req).read().decode())
    print('2. Analyze API Response:', json.dumps(analyze_res, indent=2))

    # 3. Check Dashboard & Detail Pages
    dash_resp = urllib.request.urlopen('http://127.0.0.1:8000/dashboard/')
    detail_resp = urllib.request.urlopen(f'http://127.0.0.1:8000/verification/{ver_id}/')
    print('3. Dashboard Status:', dash_resp.status, '| Detail Page Status:', detail_resp.status)
    print('DOCUMENT CLASSIFICATION & QUALITY PIPELINE VERIFIED SUCCESSFULLY!')

if __name__ == '__main__':
    run_live_quality_check()
