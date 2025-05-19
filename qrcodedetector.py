import cv2

def check_and_read_qr_code(image_path):
    """
    Checks if a QR code exists in the image and returns the decoded data if present.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: (has_qr: bool, data: str or None)
    """
    image = cv2.imread(image_path)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image)

    if points is not None and data:
        return True, data
    else:
        return False, None

image_path = 'nonqrcode.png'
has_qr, qr_data = check_and_read_qr_code(image_path)
print("QR Found:", has_qr)
if has_qr:
    print("QR Data:", qr_data)
