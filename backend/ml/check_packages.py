import pkg_resources
installed = [pkg.key for pkg in pkg_resources.working_set]
print("Installed packages:", [p for p in installed if any(x in p for x in ['face', 'vision', 'torch', 'cv', 'image', 'pil', 'ocr', 'pad'])])
